//! DuckDB loadable extension: IVF-Flat approximate nearest-neighbor vector search.
//!
//! Pure-Rust IVF index (see [`ivf`]) exposed through the quack-rs C Extension API.
//! No ANN index exists in DuckDB core (only brute-force `list_distance`), so this
//! is the sublinear-ANN gap; it complements the `fts` extension's BM25 (hybrid =
//! RRF over both, see the companion `hybrid_search.sql` macro layer).
//!
//! Function surface:
//!
//!   ivf_create_index(name, table, id_column, vector_column, nlist, niter, normalize)
//!       -> BIGINT   (rows indexed; trains an IVF index over `vector_column`)
//!   ivf_search(name, query_table, query_column, top_k, nprobe)
//!       -> table(id BIGINT, score DOUBLE)   (ANN: probe nprobe nearest lists)
//!   ivf_search_exact(name, query_table, query_column, top_k)
//!       -> table(id BIGINT, score DOUBLE)   (brute force; recall baseline)
//!   ivf_hybrid_search(name, table, id_column, text_column, query,
//!                     query_table, query_column,
//!                     [top_k, per_path, nprobe, k_rrf])
//!       -> table(id BIGINT, score DOUBLE, bm25_rank INT, vec_rank INT)
//!       FTS BM25 (fts extension) + IVF vector, fused by Reciprocal Rank
//!       Fusion; ties broken by id for reproducible results. The query
//!       vector is read from a single-row table (query_table.query_column
//!       = the query's embedding). This is the in-process hybrid-search
//!       entry point that complements `fts`. id_column may be empty.
//!   ivf_list()
//!       -> table(name VARCHAR, dim INT, n BIGINT, nlist INT)
//!   ivf_version() -> VARCHAR
//!   ivf_levenshtein(a, b)            -> INTEGER  (OSA edit distance, saturated at 3)
//!   ivf_levenshtein(a, b, maxd)      -> INTEGER  (saturated at maxd)
//!
//! `ivf_levenshtein` is the building block for Meilisearch-style typo
//! tolerance on top of the `fts` extension: the macro layer in
//! `test/fts_typo.sql` expands a query word to its closest
//! `fts_main_<T>.raw_dict` term within a length-tiered edit budget
//! (transpositions cost 1, so "sahcin"->"sachin" is corrected where plain
//! Levenshtein would need 2).
//!
//! Conventions:
//!   * `vector_column` must be `FLOAT[d]` (a LIST of FLOAT). `normalize=true`
//!     L2-normalizes every vector at build time, so score = cosine similarity.
//!   * the query vector is read from a single-row table (`query_table`) via
//!     `query_column`; it must have the same dimension as the index.
//!   * `nprobe == nlist` makes `ivf_search` exhaustive (recall 1.0); smaller
//!     `nprobe` trades recall for speed.
//!
//! C-API constraint (inherited from every quack-rs extension): table/scalar
//! callbacks do NOT receive the caller's connection. The extension opens one
//! dedicated connection from the FIRST database that LOADs it and runs its
//! `SELECT ... FROM {table}` there. Consequences: temporary tables and
//! uncommitted transactions are invisible; exactly one database per process
//! (a second, different database makes every call fail loudly).

use std::cell::Cell;
use std::collections::HashMap;
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard, OnceLock};

use libduckdb_sys::duckdb_vector_size;
use quack_rs::connection::{Connection, Registrar};
use quack_rs::data_chunk::DataChunk;
use quack_rs::error::ExtensionError;
use quack_rs::query::OwnedConnection;
use quack_rs::scalar::{ScalarFunctionBuilder, ScalarFunctionInfo, ScalarFunctionSetBuilder, ScalarOverloadBuilder};
use quack_rs::table::{BindInfo, FfiBindData, FfiInitData, InitInfo, TableFunctionBuilder};
use quack_rs::types::TypeId;
use quack_rs::vector::complex::ListVector;
use quack_rs::vector::{VectorReader, VectorWriter};

mod ivf;
use ivf::Ivf;
mod lev;

const VERSION: &str = "0.1.0";
const DEFAULT_SEED: u64 = 0x5EED_5EED_0BAD_5EED;

// ─────────────────────────────────────────────────────────────────────────────
// Global index registry (process lifetime, addressed by name)
// ─────────────────────────────────────────────────────────────────────────────

pub struct VssIndex {
    pub ivf: Ivf,
    /// The caller's id for each dense row (aligned with `ivf.data`).
    pub ids: Vec<i64>,
}

static INDEXES: LazyLock<Mutex<HashMap<String, VssIndex>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// ─────────────────────────────────────────────────────────────────────────────
// Session connection (single-database, re-entry safe)
// ─────────────────────────────────────────────────────────────────────────────

static SESSION: OnceLock<Mutex<OwnedConnection>> = OnceLock::new();
static FIRST_DB: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static MULTI_DB: AtomicBool = AtomicBool::new(false);

thread_local! {
    static IN_QUERY: Cell<bool> = const { Cell::new(false) };
}

enum SessionError {
    Reentrant,
    MultiDb,
    Unavailable,
}

impl SessionError {
    fn message(&self) -> &'static str {
        match self {
            Self::Reentrant => {
                "ivf: nested ivf call detected (a view or table function is calling ivf \
                 while another ivf call is executing on this thread); not supported"
            }
            Self::MultiDb => {
                "ivf: the extension was loaded on more than one database in this process; \
                 v0.1 supports a single database instance"
            }
            Self::Unavailable => "ivf: the extension session is unavailable",
        }
    }
}

struct ReentryMark;
impl ReentryMark {
    fn new() -> Self {
        IN_QUERY.with(|c| c.set(true));
        ReentryMark
    }
}
impl Drop for ReentryMark {
    fn drop(&mut self) {
        IN_QUERY.with(|c| c.set(false));
    }
}

struct SessionConn {
    con: MutexGuard<'static, OwnedConnection>,
    _mark: ReentryMark,
}

fn session() -> Result<SessionConn, SessionError> {
    if IN_QUERY.with(|c| c.get()) {
        return Err(SessionError::Reentrant);
    }
    if MULTI_DB.load(Ordering::SeqCst) {
        return Err(SessionError::MultiDb);
    }
    let m = SESSION.get().ok_or(SessionError::Unavailable)?;
    let g = match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    Ok(SessionConn { con: g, _mark: ReentryMark::new() })
}

/// Conservative identifier check (plain identifiers, optional `schema.` prefix).
/// Table/column names are interpolated into SQL text, so this is the injection
/// guard.
fn is_safe_identifier(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() > 2 {
        return false;
    }
    for p in &parts {
        if p.is_empty() {
            return false;
        }
        let bytes = p.as_bytes();
        if !bytes[0].is_ascii_alphabetic() && bytes[0] != b'_' {
            return false;
        }
        if !bytes.iter().all(|b| b.is_ascii_alphanumeric() || *b == b'_') {
            return false;
        }
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Data-plane helpers: read a FLOAT[d] LIST column into Rust
// ─────────────────────────────────────────────────────────────────────────────

/// Read row `row` of column `col` (a LIST(FLOAT)) from `chunk` into `out`.
/// Returns false if the row's list is NULL.
unsafe fn read_list_f32(chunk: &DataChunk, col: usize, row: usize, out: &mut Vec<f32>) -> bool {
    let reader = chunk.reader(col);
    if !reader.is_valid(row) {
        return false;
    }
    let vec = chunk.vector(col);
    let total = ListVector::get_size(vec) as usize;
    let entry = ListVector::get_entry(vec, row);
    let offset = entry.offset as usize;
    let length = entry.length as usize;
    let child = ListVector::child_reader(vec, total);
    out.clear();
    out.reserve(length);
    for d in 0..length {
        out.push(child.read_f32(offset + d));
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Table function: ivf_create_index(name, table, id_column, vector_column, nlist, niter, normalize)
//   -> (rows_indexed BIGINT)
//
// A TABLE function, not a scalar: this is a side-effectful function that runs a nested
// query and mutates the global index registry. It must not be a pure scalar, because
// DuckDB's optimizer constant-folds all-constant scalar calls during the optimize pass,
// and that isolated fold context is where the nested-query LIST read segfaulted. Table
// functions run their init in a clean execution context and are never constant-folded —
// the same model tantivy uses for its index-building path.
// ─────────────────────────────────────────────────────────────────────────────

struct CreateBind {
    name: String,
    table: String,
    id_col: String,
    vec_col: String,
    nlist: usize,
    niter: usize,
    normalize: bool,
}

struct CreateState {
    rows: i64,
    done: bool,
}

/// Build and register the IVF index for one call. Returns the number of rows indexed.
fn do_create_index(b: &CreateBind) -> Result<i64, String> {
    let sess = session().map_err(|e| e.message().to_string())?;
    // Materialize a variable-length LIST(FLOAT):
    //  1. fixed-size ARRAY (FLOAT[N]) -> LIST: duckdb_list_vector_get_size segfaults
    //     on an array vector, so the C-API list read needs a real LIST.
    //  2. element type -> FLOAT, PHYSICALLY: `vec::FLOAT[]` on a DOUBLE[] only rewrites
    //     the declared type — the child buffer can stay FLOAT8, and the C-API child
    //     reader (read_f32) then walks it half-aligned and returns garbage (no crash).
    //     list_transform forces elementwise conversion to FLOAT (verified).
    let norm = format!("list_transform({}::FLOAT[], lambda x: x::FLOAT)", b.vec_col);
    let sel = if b.id_col.is_empty() {
        format!("SELECT row_number() OVER () AS __id, {norm} AS __v FROM {}", b.table)
    } else {
        format!("SELECT {}::BIGINT AS __id, {norm} AS __v FROM {}", b.id_col, b.table)
    };
    let mut data: Vec<f32> = Vec::new();
    let mut ids: Vec<i64> = Vec::new();
    let mut dim: Option<usize> = None;
    let mut result = sess.con.query(&sel).map_err(|e| e.to_string())?;
    let mut vbuf: Vec<f32> = Vec::new();
    loop {
        match result.next_chunk() {
            Ok(Some(chunk)) => {
                let n = chunk.size();
                if n == 0 {
                    continue;
                }
                // Read the LIST(FLOAT) column with a child_reader built ONCE per chunk
                // (quack's proven pattern, tests/ffi_roundtrip/arrow_import.rs); building
                // it per row desyncs the child cursor.
                let list_vec = unsafe { chunk.vector(1) };
                let total = unsafe { ListVector::get_size(list_vec) };
                let child = unsafe { ListVector::child_reader(list_vec, total) };
                let id_reader = unsafe { chunk.reader(0) };
                let vec_reader = unsafe { chunk.reader(1) };
                for row in 0..n {
                    if !unsafe { vec_reader.is_valid(row) } {
                        continue; // NULL vector -> skip row
                    }
                    let entry = unsafe { ListVector::get_entry(list_vec, row) };
                    let off = entry.offset as usize;
                    let len = entry.length as usize;
                    vbuf.clear();
                    vbuf.reserve(len);
                    for d in 0..len {
                        vbuf.push(unsafe { child.read_f32(off + d) });
                    }
                    match dim {
                        None => dim = Some(len),
                        Some(dm) if len != dm => {
                            return Err(format!(
                                "ivf_create_index: ragged vectors (dim {} vs {}); use a fixed-length FLOAT[{}]",
                                len, dm, dm
                            ));
                        }
                        _ => {}
                    }
                    ids.push(unsafe { id_reader.read_i64(row) });
                    data.extend_from_slice(&vbuf);
                }
            }
            Ok(None) => break,
            Err(e) => return Err(format!("ivf_create_index: {e}")),
        }
    }
    let n = ids.len();
    if n == 0 {
        return Err("ivf_create_index: no rows to index".into());
    }
    let d = dim.ok_or("ivf_create_index: no vectors")?;
    if d == 0 {
        return Err("ivf_create_index: first vector is empty".into());
    }
    let default_nlist = (n as f64).sqrt() as usize;
    let nlist = (if b.nlist == 0 { default_nlist } else { b.nlist }).clamp(1, 4096);
    let niter = if b.niter == 0 { 16 } else { b.niter };
    let ivf = Ivf::train(&data, d, nlist, niter, b.normalize, DEFAULT_SEED)
        .map_err(|e| format!("ivf_create_index: {e}"))?;
    let mut reg = INDEXES.lock().unwrap();
    reg.insert(b.name.clone(), VssIndex { ivf, ids });
    Ok(n as i64)
}

quack_rs::table_bind_callback!(create_index_bind, |info| {
    unsafe {
        let bind_info = BindInfo::new(info);
        let name = bind_info.get_parameter_value(0).as_str().unwrap_or_default().trim().to_string();
        let table = bind_info.get_parameter_value(1).as_str().unwrap_or_default().trim().to_string();
        let id_col = bind_info.get_parameter_value(2).as_str().unwrap_or_default().trim().to_string();
        let vec_col = bind_info.get_parameter_value(3).as_str().unwrap_or_default().trim().to_string();
        let nlist = bind_info.get_parameter_value(4).as_i64().unwrap_or(0) as usize;
        let niter = bind_info.get_parameter_value(5).as_i64().unwrap_or(0) as usize;
        let normalize = bind_info.get_parameter_value(6).as_bool().unwrap_or(true);
        if name.is_empty() {
            bind_info.set_error("ivf_create_index: name must be non-empty");
            return;
        }
        if !is_safe_identifier(&table) {
            bind_info.set_error("ivf_create_index: table must be an identifier (e.g. 'main.docs')");
            return;
        }
        if !is_safe_identifier(&vec_col) {
            bind_info.set_error("ivf_create_index: vector_column must be an identifier");
            return;
        }
        if !id_col.is_empty() && !is_safe_identifier(&id_col) {
            bind_info.set_error("ivf_create_index: id_column must be an identifier or empty");
            return;
        }
        bind_info
            .add_result_column("rows_indexed", TypeId::BigInt)
            .set_cardinality(1, true);
        FfiBindData::<CreateBind>::set(info, CreateBind {
            name, table, id_col, vec_col, nlist, niter, normalize,
        });
    }
});

quack_rs::table_init_callback!(create_index_init, |info| {
    unsafe {
        let init_info = InitInfo::new(info);
        init_info.set_max_threads(1);
        let Some(bind) = FfiBindData::<CreateBind>::get_from_init(info) else {
            FfiInitData::<CreateState>::set(info, CreateState { rows: 0, done: true });
            return;
        };
        match do_create_index(&bind) {
            Ok(rows) => FfiInitData::<CreateState>::set(info, CreateState { rows, done: false }),
            Err(e) => {
                init_info.set_error(&e);
                FfiInitData::<CreateState>::set(info, CreateState { rows: 0, done: true });
            }
        }
    }
});

quack_rs::table_scan_callback!(create_index_scan, |info, output| {
    unsafe {
        let chunk = DataChunk::from_raw(output);
        let Some(state) = FfiInitData::<CreateState>::get_mut(info) else {
            chunk.set_size(0);
            return;
        };
        if state.done {
            chunk.set_size(0);
            return;
        }
        state.done = true;
        let mut w = chunk.writer(0);
        w.write_i64(0, state.rows);
        chunk.set_size(1);
    }
});

// Shared: read the single-row query vector
// ─────────────────────────────────────────────────────────────────────────────

struct SearchBind {
    name: String,
    query_table: String,
    query_column: String,
    top_k: i64,
    nprobe: i64,
    exact: bool,
}

struct SearchState {
    hits: Vec<(i64, f32)>,
    next: usize,
}

fn read_query_vector(sess: &SessionConn, query_table: &str, query_column: &str)
    -> Result<Vec<f32>, String>
{
    // Same physical-materialization normalization as ivf_create_index: a query
    // table with DOUBLE[] (or FLOAT[N]) must be converted elementwise to
    // variable-length LIST(FLOAT), or the C-API child reader misreads f64 as f32.
    let sql = format!(
        "SELECT list_transform({query_column}::FLOAT[], lambda x: x::FLOAT) AS __q FROM {query_table} LIMIT 1"
    );
    let mut result = sess.con.query(&sql).map_err(|e| e.to_string())?;
    let mut v: Vec<f32> = Vec::new();
    match result.next_chunk() {
        Ok(Some(chunk)) if chunk.size() > 0 => {
            if !unsafe { read_list_f32(&chunk, 0, 0, &mut v) } {
                return Err("query vector is NULL".into());
            }
        }
        _ => return Err("query_table returned no rows (need exactly one row)".into()),
    }
    Ok(v)
}

/// Query vector from an in-process scalar expression (e.g.
/// `ml_embed_sentence('minilm', 'crude oil prices')`): the zg-style single-entry
/// path where the caller passes text, not a precomputed vector. Same normalization
/// as read_query_vector (variable-length LIST(FLOAT)).
fn read_query_expr(sess: &SessionConn, expr: &str) -> Result<Vec<f32>, String> {
    let sql = format!(
        "SELECT list_transform(({expr})::FLOAT[], lambda x: x::FLOAT) AS __q FROM (SELECT 1)"
    );
    let mut result = sess.con.query(&sql).map_err(|e| e.to_string())?;
    let mut v: Vec<f32> = Vec::new();
    match result.next_chunk() {
        Ok(Some(chunk)) if chunk.size() > 0 => {
            if !unsafe { read_list_f32(&chunk, 0, 0, &mut v) } {
                return Err("embed_expr returned NULL".into());
            }
        }
        _ => return Err("embed_expr produced no vector".into()),
    }
    Ok(v)
}

fn do_search(b: &SearchBind) -> Result<Vec<(i64, f32)>, String> {
    let query = {
        let sess = session().map_err(|e| e.message().to_string())?;
        read_query_vector(&sess, &b.query_table, &b.query_column)
    }?;

    let guard = INDEXES.lock().unwrap();
    let idx = guard.get(&b.name).ok_or_else(|| format!("no ivf index named '{}' — call ivf_create_index first", b.name))?;
    if idx.ivf.dim != query.len() {
        return Err(format!("query dim {} != index dim {}", query.len(), idx.ivf.dim));
    }
    let top_k = b.top_k as usize;
    let scored = if b.exact {
        idx.ivf.search_exact(&query, top_k)
    } else {
        let nprobe = if b.nprobe <= 0 { idx.ivf.nlist } else { b.nprobe as usize };
        idx.ivf.search(&query, top_k, nprobe)
    };
    Ok(scored.into_iter().map(|(dense, s)| (idx.ids[dense], s)).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Table function: ivf_search / ivf_search_exact
// ─────────────────────────────────────────────────────────────────────────────

fn bind_search(info: libduckdb_sys::duckdb_bind_info, exact: bool) {
    unsafe {
        let bind_info = BindInfo::new(info);
        let name = bind_info.get_parameter_value(0).as_str().unwrap_or_default().trim().to_string();
        let query_table = bind_info.get_parameter_value(1).as_str().unwrap_or_default().trim().to_string();
        let query_column = bind_info.get_parameter_value(2).as_str().unwrap_or_default().trim().to_string();
        let top_k = match bind_info.get_named_parameter_value("top_k").as_i64() {
            Some(v) if (1..=1_000_000).contains(&v) => v,
            Some(v) => { bind_info.set_error(&format!("ivf_search: top_k must be 1..=1000000 (got {v})")); return; }
            None => 10,
        };
        let nprobe = match bind_info.get_named_parameter_value("nprobe").as_i64() {
            Some(v) => v, // 0 or negative -> nlist (exhaustive)
            None => 0,
        };
        if !is_safe_identifier(&query_table) {
            bind_info.set_error("ivf_search: query_table must be an identifier");
            return;
        }
        if !is_safe_identifier(&query_column) {
            bind_info.set_error("ivf_search: query_column must be an identifier");
            return;
        }
        // Probe: the index must exist and the query table must be readable.
        match session() {
            Ok(sess) => {
                let guard = INDEXES.lock().unwrap();
                if !guard.contains_key(&name) {
                    bind_info.set_error(&format!("no ivf index named '{}' — call ivf_create_index first", name));
                    return;
                }
                drop(guard);
                let probe = format!("SELECT {query_column} FROM {query_table} LIMIT 0");
                if let Err(e) = sess.con.query(&probe) {
                    bind_info.set_error(&format!("ivf_search: cannot read query_table '{query_table}': {e}"));
                    return;
                }
            }
            Err(e) => { bind_info.set_error(e.message()); return; }
        }
        bind_info
            .add_result_column("id", TypeId::BigInt)
            .add_result_column("score", TypeId::Double)
            .set_cardinality(top_k as u64, false);
        FfiBindData::<SearchBind>::set(info, SearchBind {
            name, query_table, query_column, top_k, nprobe, exact,
        });
    }
}

quack_rs::table_bind_callback!(search_bind, |info| { bind_search(info, false); });
quack_rs::table_bind_callback!(search_exact_bind, |info| { bind_search(info, true); });

quack_rs::table_init_callback!(search_init, |info| {
    unsafe {
        let init_info = InitInfo::new(info);
        init_info.set_max_threads(1);
        let Some(bind) = FfiBindData::<SearchBind>::get_from_init(info) else {
            FfiInitData::<SearchState>::set(info, SearchState { hits: Vec::new(), next: 0 });
            return;
        };
        match do_search(&bind) {
            Ok(hits) => FfiInitData::<SearchState>::set(info, SearchState { hits, next: 0 }),
            Err(e) => {
                init_info.set_error(&e);
                FfiInitData::<SearchState>::set(info, SearchState { hits: Vec::new(), next: 0 });
            }
        }
    }
});

quack_rs::table_scan_callback!(search_scan, |info, output| {
    unsafe {
        let chunk = DataChunk::from_raw(output);
        let Some(state) = FfiInitData::<SearchState>::get_mut(info) else {
            chunk.set_size(0);
            return;
        };
        if state.next >= state.hits.len() {
            chunk.set_size(0);
            return;
        }
        let batch = (state.hits.len() - state.next).min(duckdb_vector_size() as usize);
        let mut id_writer = chunk.writer(0);
        let mut score_writer = chunk.writer(1);
        for i in 0..batch {
            let (id, s) = state.hits[state.next + i];
            id_writer.write_i64(i, id);
            score_writer.write_f64(i, s as f64);
        }
        state.next += batch;
        chunk.set_size(batch);
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// Table function: ivf_list()
// ─────────────────────────────────────────────────────────────────────────────

struct ListState {
    rows: Vec<(String, i32, i64, i32)>,
    next: usize,
}

quack_rs::table_bind_callback!(ivf_list_bind, |info| {
    unsafe {
        let bind_info = BindInfo::new(info);
        let n = INDEXES.lock().unwrap().len();
        bind_info
            .add_result_column("name", TypeId::Varchar)
            .add_result_column("dim", TypeId::Integer)
            .add_result_column("n", TypeId::BigInt)
            .add_result_column("nlist", TypeId::Integer)
            .set_cardinality(n as u64, true);
    }
});

quack_rs::table_init_callback!(ivf_list_init, |info| {
    unsafe {
        InitInfo::new(info).set_max_threads(1);
        let guard = INDEXES.lock().unwrap();
        let mut rows: Vec<(String, i32, i64, i32)> =
            guard.iter().map(|(k, v)| (k.clone(), v.ivf.dim as i32, v.ivf.n as i64, v.ivf.nlist as i32)).collect();
        rows.sort();
        drop(guard);
        FfiInitData::<ListState>::set(info, ListState { rows, next: 0 });
    }
});

quack_rs::table_scan_callback!(ivf_list_scan, |info, output| {
    unsafe {
        let chunk = DataChunk::from_raw(output);
        let Some(state) = FfiInitData::<ListState>::get_mut(info) else {
            chunk.set_size(0);
            return;
        };
        if state.next >= state.rows.len() {
            chunk.set_size(0);
            return;
        }
        let batch = (state.rows.len() - state.next).min(duckdb_vector_size() as usize);
        for i in 0..batch {
            let (name, dim, n, nlist) = &state.rows[state.next + i];
            let mut w = chunk.writer(0); w.write_varchar(i, name);
            let mut w = chunk.writer(1); w.write_i32(i, *dim);
            let mut w = chunk.writer(2); w.write_i64(i, *n);
            let mut w = chunk.writer(3); w.write_i32(i, *nlist);
        }
        state.next += batch;
        chunk.set_size(batch);
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// Table functions: ivf_save(name, path)  /  ivf_load(name, path)
//
// Persistence for the in-memory IVF registry. A saved index is a single binary
// file (little-endian, no compression):
//
//   [header  16 bytes: magic u32=0x4956_4643 ("FCVI" LE), version u32, dim u32, nlist u32]
//   [n  u32]
//   [n*dim   f32  data (row-major, dense order)]
//   [nlist*dim f32  centroids (row-major)]
//   [n      u32  list_of (centroid assignment per dense row)]
//   [n      i64  ids   (caller id per dense row, if has_ids; else dense 0..n)]
//
// `ivf_load` reconstructs WITHOUT retraining (k-means is the 157s cost), so a
// cold start drops from ~157s to a file read + memcpy. `lists` is rebuilt
// deterministically from `list_of`, so search results are bit-identical to the
// trained index.
// ─────────────────────────────────────────────────────────────────────────────

const IVF_MAGIC: u32 = 0x4346_5649; // "IVFC"
const IVF_VERSION: u32 = 1;

struct SaveLoadState {
    // one result row: (name, dim, n, nlist, bytes, message)
    name: String,
    dim: i32,
    n: i64,
    nlist: i32,
    bytes: i64,
    message: String,
    next: usize,
}

fn sl_row(name: &str, dim: i32, n: i64, nlist: i32, bytes: i64, message: &str) -> (String, i32, i64, i32, i64, String) {
    (name.to_string(), dim, n, nlist, bytes, message.to_string())
}

struct SlBind {
    name: String,
    path: String,
}

fn bind_sl(info: libduckdb_sys::duckdb_bind_info) {
    unsafe {
        let bind_info = BindInfo::new(info);
        let name = bind_info
            .get_parameter_value(0)
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        let path = bind_info
            .get_parameter_value(1)
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        bind_info
            .add_result_column("name", TypeId::Varchar)
            .add_result_column("dim", TypeId::Integer)
            .add_result_column("n", TypeId::BigInt)
            .add_result_column("nlist", TypeId::Integer)
            .add_result_column("bytes", TypeId::BigInt)
            .add_result_column("message", TypeId::Varchar)
            .set_cardinality(1, true);
        FfiBindData::<SlBind>::set(info, SlBind { name, path });
    }
}
quack_rs::table_bind_callback!(save_bind, |info| {
    bind_sl(info);
});
quack_rs::table_bind_callback!(load_bind, |info| {
    bind_sl(info);
});

fn read_u32(buf: &[u8], off: &mut usize) -> Option<u32> {
    if *off + 4 > buf.len() { return None; }
    let v = u32::from_le_bytes(buf[*off..*off + 4].try_into().ok()?);
    *off += 4; Some(v)
}
fn read_u32s(buf: &[u8], off: &mut usize, cnt: usize) -> Option<Vec<u32>> {
    let mut out = Vec::with_capacity(cnt);
    for _ in 0..cnt { out.push(read_u32(buf, off)?); }
    Some(out)
}
fn read_f32s(buf: &[u8], off: &mut usize, cnt: usize) -> Option<Vec<f32>> {
    if *off + cnt * 4 > buf.len() { return None; }
    let mut out = Vec::with_capacity(cnt);
    for i in 0..cnt {
        let v = f32::from_le_bytes(buf[*off + i * 4..*off + i * 4 + 4].try_into().ok()?);
        out.push(v);
    }
    *off += cnt * 4; Some(out)
}
fn read_i64s(buf: &[u8], off: &mut usize, cnt: usize) -> Option<Vec<i64>> {
    if *off + cnt * 8 > buf.len() { return None; }
    let mut out = Vec::with_capacity(cnt);
    for i in 0..cnt {
        let v = i64::from_le_bytes(buf[*off + i * 8..*off + i * 8 + 8].try_into().ok()?);
        out.push(v);
    }
    *off += cnt * 8; Some(out)
}

quack_rs::table_init_callback!(save_init, |info| {
    unsafe {
        let init_info = InitInfo::new(info);
        init_info.set_max_threads(1);
        let Some(b) = FfiBindData::<SlBind>::get_from_init(info) else {
            FfiInitData::<SaveLoadState>::set(
                info,
                SaveLoadState { name: String::new(), dim: 0, n: 0, nlist: 0, bytes: 0, message: "no bind data".into(), next: 0 },
            );
            return;
        };
        let name = b.name.clone();
        let path = b.path.clone();

        let (name_o, dim, n, nlist, bytes, message) = if name.is_empty() || path.is_empty() {
            sl_row(&name, 0, 0, 0, 0, "ivf_save: name and path must be non-empty")
        } else {
            let guard = INDEXES.lock().unwrap();
            match guard.get(&name) {
                Some(v) => {
                    let ivf = &v.ivf;
                    let n = ivf.n;
                    let dim = ivf.dim;
                    let nlist = ivf.nlist;
                    let ids_len = if v.ids.len() == n { n } else { 0 };
                    let mut buf: Vec<u8> = Vec::with_capacity(
                        20 + n * 4 + n * dim * 4 + nlist * dim * 4 + n * 4 + ids_len * 8);
                    // header
                    buf.extend_from_slice(&IVF_MAGIC.to_le_bytes());
                    buf.extend_from_slice(&IVF_VERSION.to_le_bytes());
                    buf.extend_from_slice(&(dim as u32).to_le_bytes());
                    buf.extend_from_slice(&(nlist as u32).to_le_bytes());
                    buf.extend_from_slice(&(n as u32).to_le_bytes());
                    // data
                    for &x in ivf.data.iter() { buf.extend_from_slice(&x.to_le_bytes()); }
                    // centroids
                    for &x in ivf.centroids.iter() { buf.extend_from_slice(&x.to_le_bytes()); }
                    // list_of
                    for &c in ivf.list_of.iter() { buf.extend_from_slice(&(c as u32).to_le_bytes()); }
                    // ids (if aligned)
                    if ids_len > 0 { for &id in v.ids.iter() { buf.extend_from_slice(&id.to_le_bytes()); } }
                    match fs::write(&path, &buf) {
                        Ok(_) => sl_row(&name, dim as i32, n as i64, nlist as i32, buf.len() as i64, "ok"),
                        Err(e) => sl_row(&name, dim as i32, n as i64, nlist as i32, 0, &format!("write failed: {e}")),
                    }
                }
                None => sl_row(&name, 0, 0, 0, 0, "no such index — call ivf_create_index or ivf_load first"),
            }
        };
        FfiInitData::<SaveLoadState>::set(
            info,
            SaveLoadState { name: name_o, dim, n, nlist, bytes, message, next: 0 },
        );
    }
});

quack_rs::table_init_callback!(load_init, |info| {
    unsafe {
        let init_info = InitInfo::new(info);
        init_info.set_max_threads(1);
        let Some(b) = FfiBindData::<SlBind>::get_from_init(info) else {
            FfiInitData::<SaveLoadState>::set(
                info,
                SaveLoadState { name: String::new(), dim: 0, n: 0, nlist: 0, bytes: 0, message: "no bind data".into(), next: 0 },
            );
            return;
        };
        let name = b.name.clone();
        let path = b.path.clone();

        let (name_o, dim, n, nlist, bytes, message) = if name.is_empty() || path.is_empty() {
            sl_row(&name, 0, 0, 0, 0, "ivf_load: name and path must be non-empty")
        } else {
            match fs::read(&path) {
                Err(e) => sl_row(&name, 0, 0, 0, 0, &format!("read failed: {e}")),
                Ok(buf) => {
                    let mut off = 0usize;
                    let magic = read_u32(&buf, &mut off).unwrap_or(0);
                    let version = read_u32(&buf, &mut off).unwrap_or(0);
                    if magic != IVF_MAGIC {
                        sl_row(&name, 0, 0, 0, buf.len() as i64, "bad magic — not an ivf file")
                    } else if version != IVF_VERSION {
                        sl_row(&name, 0, 0, 0, buf.len() as i64, &format!("unsupported version {version}"))
                    } else {
                        let dim = read_u32(&buf, &mut off).map(|v| v as usize).unwrap_or(0);
                        let nlist = read_u32(&buf, &mut off).map(|v| v as usize).unwrap_or(0);
                        let n = read_u32(&buf, &mut off).map(|v| v as usize).unwrap_or(0);
                        let data = read_f32s(&buf, &mut off, n.saturating_mul(dim));
                        let cents = read_f32s(&buf, &mut off, nlist.saturating_mul(dim));
                        let list_of = read_u32s(&buf, &mut off, n).map(|v| v.iter().map(|&x| x as usize).collect::<Vec<usize>>());
                        let ids = read_i64s(&buf, &mut off, n);
                        let (data, cents, list_of, ids) = match (data, cents, list_of, ids) {
                            (Some(d), Some(c), Some(l), Some(i)) => (d, c, l, i),
                            _ => {
                                FfiInitData::<SaveLoadState>::set(
                                    info,
                                    SaveLoadState { name: name.clone(), dim: 0, n: 0, nlist: 0, bytes: buf.len() as i64, message: "truncated file".into(), next: 0 },
                                );
                                return;
                            }
                        };
                        let ids = if ids.len() == n { ids } else { (0..n as i64).collect::<Vec<i64>>() };
                        match Ivf::load_from(dim, n, nlist, cents, list_of, data) {
                            Err(e) => sl_row(&name, dim as i32, n as i64, nlist as i32, buf.len() as i64, &e),
                            Ok(ivf) => {
                                INDEXES.lock().unwrap().insert(name.clone(), VssIndex { ivf, ids });
                                sl_row(&name, dim as i32, n as i64, nlist as i32, buf.len() as i64, "ok (no retrain)")
                            }
                        }
                    }
                }
            }
        };
        FfiInitData::<SaveLoadState>::set(
            info,
            SaveLoadState { name: name_o, dim, n, nlist, bytes, message, next: 0 },
        );
    }
});

quack_rs::table_scan_callback!(save_scan, |info, output| {
    unsafe {
        let chunk = DataChunk::from_raw(output);
        let Some(state) = FfiInitData::<SaveLoadState>::get_mut(info) else {
            chunk.set_size(0);
            return;
        };
        if state.next >= 1 {
            chunk.set_size(0);
            return;
        }
        let mut w = chunk.writer(0);
        w.write_varchar(0, &state.name);
        let mut w = chunk.writer(1);
        w.write_i32(0, state.dim);
        let mut w = chunk.writer(2);
        w.write_i64(0, state.n);
        let mut w = chunk.writer(3);
        w.write_i32(0, state.nlist);
        let mut w = chunk.writer(4);
        w.write_i64(0, state.bytes);
        let mut w = chunk.writer(5);
        w.write_varchar(0, &state.message);
        state.next = 1;
        chunk.set_size(1);
    }
});
quack_rs::table_scan_callback!(load_scan, |info, output| {
    unsafe {
        let chunk = DataChunk::from_raw(output);
        let Some(state) = FfiInitData::<SaveLoadState>::get_mut(info) else {
            chunk.set_size(0);
            return;
        };
        if state.next >= 1 {
            chunk.set_size(0);
            return;
        }
        let mut w = chunk.writer(0);
        w.write_varchar(0, &state.name);
        let mut w = chunk.writer(1);
        w.write_i32(0, state.dim);
        let mut w = chunk.writer(2);
        w.write_i64(0, state.n);
        let mut w = chunk.writer(3);
        w.write_i32(0, state.nlist);
        let mut w = chunk.writer(4);
        w.write_i64(0, state.bytes);
        let mut w = chunk.writer(5);
        w.write_varchar(0, &state.message);
        state.next = 1;
        chunk.set_size(1);
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// Table function: ivf_hybrid_search — FTS BM25 + IVF vector, RRF-fused.
// ─────────────────────────────────────────────────────────────────────────────

struct HybridBind {
    name: String,
    table: String,
    id_col: String,
    text_col: String,
    query: String,
    query_table: String,
    query_col: String,
    top_k: i64,
    per_path: i64,
    nprobe: i64,
    k_rrf: i64,
    embed_expr: String,
}

struct HybridRow {
    id: i64,
    rrf: f64,
    bm25_rank: i64,
    vec_rank: i64,
}

struct HybridState {
    rows: Vec<HybridRow>,
    next: usize,
}

fn do_hybrid(b: &HybridBind) -> Result<Vec<HybridRow>, String> {
    // Vector path: query vector from embed_expr (in-process scalar, e.g.
    // ml_embed_sentence(...)) if given, else from the single-row query table.
    let vec_hits: Vec<(i64, usize)> = {
        let query = {
            let sess = session().map_err(|e| e.message().to_string())?;
            if b.embed_expr.is_empty() {
                read_query_vector(&sess, &b.query_table, &b.query_col)
            } else {
                read_query_expr(&sess, &b.embed_expr)
            }
        }?;
        let guard = INDEXES.lock().unwrap();
        let idx = guard
            .get(&b.name)
            .ok_or_else(|| format!("no ivf index named '{}' — call ivf_create_index or ivf_load first", b.name))?;
        if idx.ivf.dim != query.len() {
            return Err(format!("query dim {} != index dim {}", query.len(), idx.ivf.dim));
        }
        let nprobe = if b.nprobe <= 0 { idx.ivf.nlist } else { b.nprobe as usize };
        idx.ivf
            .search(&query, b.per_path as usize, nprobe)
            .into_iter()
            .enumerate()
            .map(|(r, (dense, _))| (idx.ids[dense], r))
            .collect()
    };
    // Text path: FTS layered BM25 on the table's text column. Graceful: if the
    // table has no FTS index (fts_main_{table} missing), degrade to vector-only
    // (fts_hits empty) instead of erroring — the RRF then just scores the vector
    // arm, so a table without FTS still works.
    let q_lit = format!("'{}'", b.query.replace('\'', "''"));
    let fts_sql = format!(
        "SELECT docname::BIGINT AS id, rank FROM fts_main_{}.search_layered_bm25({}, top_k := {})",
        b.table, q_lit, b.per_path
    );
    let fts_hits: Vec<(i64, usize)> = {
        let sess = session().map_err(|e| e.message().to_string())?;
        // Probe FTS schema existence first (search_layered_bm25 is only present
        // when the fts extension built fts_main_{table} for this table).
        let probe = format!(
            "SELECT 1 FROM fts_main_{}.search_layered_bm25({}, top_k := 1) LIMIT 1",
            b.table, q_lit
        );
        if sess.con.query(&probe).is_err() {
            Vec::new()
        } else {
            let mut res = sess
                .con
                .query(&fts_sql)
                .map_err(|e| format!("ivf_hybrid_search: fts: {e}"))?;
            let mut out = Vec::new();
            let mut r = 0usize;
            loop {
                match res.next_chunk() {
                    Ok(Some(chunk)) if chunk.size() > 0 => {
                        let n = chunk.size();
                        let ir = unsafe { chunk.reader(0) };
                        for i in 0..n {
                            out.push((unsafe { ir.read_i64(i) }, r));
                            r += 1;
                        }
                    }
                    Ok(Some(_)) | Ok(None) => break,
                    Err(e) => return Err(format!("ivf_hybrid_search: fts: {e}")),
                }
            }
            out
        }
    };
    // Reciprocal Rank Fusion (rank is 1-based).
    let mut rrf: HashMap<i64, f64> = HashMap::new();
    let mut b25: HashMap<i64, i64> = HashMap::new();
    let mut vrank: HashMap<i64, i64> = HashMap::new();
    for (id, r) in &fts_hits {
        *b25.entry(*id).or_insert(0) += 1;
        *rrf.entry(*id).or_insert(0.0) += 1.0 / (b.k_rrf as f64 + (*r as f64) + 1.0);
    }
    for (id, r) in &vec_hits {
        let cur = vrank.entry(*id).or_insert(0);
        *cur += 1;
        *rrf.entry(*id).or_insert(0.0) += 1.0 / (b.k_rrf as f64 + (*r as f64) + 1.0);
    }
    let mut rows: Vec<HybridRow> = rrf
        .into_iter()
        .map(|(id, s)| HybridRow {
            id,
            rrf: s,
            bm25_rank: b25.get(&id).copied().unwrap_or(0) as i64,
            vec_rank: vrank.get(&id).copied().unwrap_or(0) as i64,
        })
        .collect();
    rows.sort_by(|a, b| b.rrf.partial_cmp(&a.rrf).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.id.cmp(&b.id)));
    rows.truncate(b.top_k as usize);
    Ok(rows)
}

quack_rs::table_bind_callback!(hybrid_bind, |info| {
    unsafe {
        let bind_info = BindInfo::new(info);
        let name = bind_info
            .get_parameter_value(0)
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        let table = bind_info
            .get_parameter_value(1)
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        let id_col = bind_info
            .get_parameter_value(2)
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        let text_col = bind_info
            .get_parameter_value(3)
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        let query = bind_info
            .get_parameter_value(4)
            .as_str()
            .unwrap_or_default()
            .to_string();
        let query_table = bind_info
            .get_parameter_value(5)
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        let query_col = bind_info
            .get_parameter_value(6)
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        for (lbl, s) in [
            ("table", &table),
            ("text_col", &text_col),
            ("query_table", &query_table),
            ("query_col", &query_col),
        ] {
            if !is_safe_identifier(s) {
                bind_info.set_error(&format!("ivf_hybrid_search: {lbl} must be an identifier"));
                return;
            }
        }
        if !id_col.is_empty() && !is_safe_identifier(&id_col) {
            bind_info.set_error("ivf_hybrid_search: id_col must be an identifier or empty");
            return;
        }
        let top_k = bind_info.get_named_parameter_value("top_k").as_i64().unwrap_or(10);
        let per_path = bind_info.get_named_parameter_value("per_path").as_i64().unwrap_or(50);
        let nprobe = bind_info.get_named_parameter_value("nprobe").as_i64().unwrap_or(0);
        let k_rrf = bind_info.get_named_parameter_value("k_rrf").as_i64().unwrap_or(60);
        if name.is_empty() || table.is_empty() || text_col.is_empty() || query_table.is_empty() {
            bind_info.set_error("ivf_hybrid_search: name, table, text_col, query_table are required");
            return;
        }
        // bind-time existence check (index + query table) so callers get a clear error.
        if let Ok(sess) = session() {
            {
                let guard = INDEXES.lock().unwrap();
                if !guard.contains_key(&name) {
                    bind_info.set_error(&format!(
                        "no ivf index named '{}' — call ivf_create_index or ivf_load first",
                        name
                    ));
                    return;
                }
            }
            let probe2 = format!("SELECT {} FROM {} LIMIT 0", query_col, query_table);
            if let Err(e) = sess.con.query(&probe2) {
                bind_info.set_error(&format!(
                    "ivf_hybrid_search: cannot read query vector from {query_table}.{query_col}: {e}"
                ));
                return;
            }
        }
        bind_info
            .add_result_column("id", TypeId::BigInt)
            .add_result_column("score", TypeId::Double)
            .add_result_column("bm25_rank", TypeId::BigInt)
            .add_result_column("vec_rank", TypeId::BigInt)
            .set_cardinality(top_k.max(1) as u64, false);
        FfiBindData::<HybridBind>::set(
            info,
            HybridBind {
                name,
                table,
                id_col,
                text_col,
                query,
                query_table,
                query_col,
                top_k,
                per_path,
                nprobe,
                k_rrf,
                embed_expr: String::new(),
            },
        );
    }
});

/// Bind for `ivf_hybrid_embed(name, table, id_col, text_col, query, embed_expr)`:
/// the single-call, text-in entry point. `embed_expr` is an in-process scalar SQL
/// expression that produces the query vector, e.g.
/// `ml_embed_sentence('minilm', 'crude oil')` — no precomputed query table needed.
/// `query_table`/`query_col` are left empty so `do_hybrid` uses the embed path.
quack_rs::table_bind_callback!(embed_bind, |info| {
    unsafe {
        let bind_info = BindInfo::new(info);
        let name = bind_info
            .get_parameter_value(0)
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        let table = bind_info
            .get_parameter_value(1)
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        let id_col = bind_info
            .get_parameter_value(2)
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        let text_col = bind_info
            .get_parameter_value(3)
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        let query = bind_info
            .get_parameter_value(4)
            .as_str()
            .unwrap_or_default()
            .to_string();
        let embed_expr = bind_info
            .get_parameter_value(5)
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();
        for (lbl, s) in [("table", &table), ("text_col", &text_col)] {
            if !is_safe_identifier(s) {
                bind_info.set_error(&format!("ivf_hybrid_embed: {lbl} must be an identifier"));
                return;
            }
        }
        if !id_col.is_empty() && !is_safe_identifier(&id_col) {
            bind_info.set_error("ivf_hybrid_embed: id_col must be an identifier or empty");
            return;
        }
        let top_k = bind_info
            .get_named_parameter_value("top_k")
            .as_i64()
            .unwrap_or(10);
        let per_path = bind_info
            .get_named_parameter_value("per_path")
            .as_i64()
            .unwrap_or(50);
        let nprobe = bind_info
            .get_named_parameter_value("nprobe")
            .as_i64()
            .unwrap_or(0);
        let k_rrf = bind_info
            .get_named_parameter_value("k_rrf")
            .as_i64()
            .unwrap_or(60);
        if name.is_empty() || table.is_empty() || text_col.is_empty() || embed_expr.is_empty() {
            bind_info.set_error("ivf_hybrid_embed: name, table, text_col, embed_expr are required");
            return;
        }
        // bind-time index-existence check for a clear error.
        {
            let guard = INDEXES.lock().unwrap();
            if !guard.contains_key(&name) {
                bind_info.set_error(&format!(
                    "no ivf index named '{}' — call ivf_create_index or ivf_load first",
                    name
                ));
                return;
            }
        }
        bind_info
            .add_result_column("id", TypeId::BigInt)
            .add_result_column("score", TypeId::Double)
            .add_result_column("bm25_rank", TypeId::BigInt)
            .add_result_column("vec_rank", TypeId::BigInt)
            .set_cardinality(top_k.max(1) as u64, false);
        FfiBindData::<HybridBind>::set(
            info,
            HybridBind {
                name,
                table,
                id_col,
                text_col,
                query,
                query_table: String::new(),
                query_col: String::new(),
                top_k,
                per_path,
                nprobe,
                k_rrf,
                embed_expr,
            },
        );
    }
});

quack_rs::table_init_callback!(hybrid_init, |info| {
    unsafe {
        let init_info = InitInfo::new(info);
        init_info.set_max_threads(1);
        let Some(bind) = FfiBindData::<HybridBind>::get_from_init(info) else {
            FfiInitData::<HybridState>::set(
                info,
                HybridState {
                    rows: Vec::new(),
                    next: 0,
                },
            );
            return;
        };
        match do_hybrid(&bind) {
            Ok(rows) => {
                FfiInitData::<HybridState>::set(
                    info,
                    HybridState {
                        rows,
                        next: 0,
                    },
                );
            }
            Err(e) => {
                init_info.set_error(&e);
                FfiInitData::<HybridState>::set(
                    info,
                    HybridState {
                        rows: Vec::new(),
                        next: 0,
                    },
                );
            }
        }
    }
});

quack_rs::table_scan_callback!(hybrid_scan, |info, output| {
    unsafe {
        let chunk = DataChunk::from_raw(output);
        let Some(state) = FfiInitData::<HybridState>::get_mut(info) else {
            chunk.set_size(0);
            return;
        };
        if state.next >= state.rows.len() {
            chunk.set_size(0);
            return;
        }
        let batch = (state.rows.len() - state.next).min(duckdb_vector_size() as usize);
        for i in 0..batch {
            let r = &state.rows[state.next + i];
            let mut w = chunk.writer(0);
            w.write_i64(i, r.id);
            let mut w = chunk.writer(1);
            w.write_f64(i, r.rrf);
            let mut w = chunk.writer(2);
            w.write_i64(i, r.bm25_rank);
            let mut w = chunk.writer(3);
            w.write_i64(i, r.vec_rank);
        }
        state.next += batch;
        chunk.set_size(batch);
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// Scalar: ivf_version()
// ─────────────────────────────────────────────────────────────────────────────

quack_rs::scalar_callback!(version_fn, |_info, _input, output| {
    unsafe {
        let mut writer = VectorWriter::new(output);
        writer.write_varchar(0, VERSION);
    }
});

// ivf_levenshtein(a, b) — saturated at 3 (typo tiers: 0 exact / 1 / 2 / >2)
quack_rs::scalar_callback!(lev_2_fn, |info, input, output| {
    unsafe {
        let a = VectorReader::new(input, 0);
        let b = VectorReader::new(input, 1);
        let n = a.row_count();
        let mut writer = VectorWriter::new(output);
        for row in 0..n {
            if !a.is_valid(row) || !b.is_valid(row) {
                writer.set_null(row);
                continue;
            }
            let d = lev::levenshtein_bounded(a.read_str(row), b.read_str(row), 3);
            writer.write_i32(row, d as i32);
        }
        let _ = info;
    }
});

// ivf_levenshtein(a, b, maxd) — caller-chosen saturation
quack_rs::scalar_callback!(lev_3_fn, |info, input, output| {
    unsafe {
        let a = VectorReader::new(input, 0);
        let b = VectorReader::new(input, 1);
        let md = VectorReader::new(input, 2);
        let n = a.row_count();
        let mut writer = VectorWriter::new(output);
        for row in 0..n {
            if !a.is_valid(row) || !b.is_valid(row) || !md.is_valid(row) {
                writer.set_null(row);
                continue;
            }
            let maxd = (md.read_i32(row) as usize).max(1).min(64);
            let d = lev::levenshtein_bounded(a.read_str(row), b.read_str(row), maxd);
            writer.write_i32(row, d as i32);
        }
        let _ = info;
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────────────

unsafe fn register_all(con: &Connection) -> Result<(), ExtensionError> {
    // Bind the session connection to the first database that loads us.
    let db = con.as_raw_database() as *mut ();
    match FIRST_DB.load(Ordering::SeqCst) {
        null if null.is_null() => {
            let won = FIRST_DB.compare_exchange(std::ptr::null_mut(), db, Ordering::SeqCst, Ordering::SeqCst);
            if won.is_ok() {
                let owned = con.open_connection()?;
                let _ = SESSION.set(Mutex::new(owned));
            } else {
                let first = won.unwrap_err();
                if db != first && !db.is_null() {
                    MULTI_DB.store(true, Ordering::SeqCst);
                }
            }
        }
        first => {
            if db != first && !db.is_null() {
                MULTI_DB.store(true, Ordering::SeqCst);
            }
        }
    }

    con.register_table(
        TableFunctionBuilder::new("ivf_create_index")
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .param(TypeId::BigInt)
            .param(TypeId::BigInt)
            .param(TypeId::Boolean)
            .bind(create_index_bind)
            .init(create_index_init)
            .scan(create_index_scan),
    )?;

    con.register_table(
        TableFunctionBuilder::new("ivf_search")
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .named_param("top_k", TypeId::BigInt)
            .named_param("nprobe", TypeId::BigInt)
            .bind(search_bind)
            .init(search_init)
            .scan(search_scan),
    )?;

    con.register_table(
        TableFunctionBuilder::new("ivf_search_exact")
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .named_param("top_k", TypeId::BigInt)
            .bind(search_exact_bind)
            .init(search_init)
            .scan(search_scan),
    )?;

    con.register_table(
        TableFunctionBuilder::new("ivf_list")
            .bind(ivf_list_bind)
            .init(ivf_list_init)
            .scan(ivf_list_scan),
    )?;

    con.register_table(
        TableFunctionBuilder::new("ivf_save")
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .bind(save_bind)
            .init(save_init)
            .scan(save_scan),
    )?;

    con.register_table(
        TableFunctionBuilder::new("ivf_load")
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .bind(load_bind)
            .init(load_init)
            .scan(load_scan),
    )?;

    con.register_table(
        TableFunctionBuilder::new("ivf_hybrid_search")
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .named_param("top_k", TypeId::BigInt)
            .named_param("per_path", TypeId::BigInt)
            .named_param("nprobe", TypeId::BigInt)
            .named_param("k_rrf", TypeId::BigInt)
            .bind(hybrid_bind)
            .init(hybrid_init)
            .scan(hybrid_scan),
    )?;

    // ivf_hybrid_embed(name, table, id_col, text_col, query, embed_expr)
    // -> table(id BIGINT, score DOUBLE, bm25_rank BIGINT, vec_rank BIGINT)
    //
    // The single-call, text-in entry point. `embed_expr` is an in-process scalar
    // SQL expression that produces the query vector, e.g.
    //   ivf_hybrid_embed('news','news','id','text','crude oil',
    //                    "ml_embed_sentence('minilm','crude oil')")
    // so no precomputed query table is needed — the vector is embedded in-process
    // (duckdb-ml) and fused with FTS BM25 via RRF. Named params (top_k, per_path,
    // nprobe, k_rrf) and result columns are shared with ivf_hybrid_search.
    con.register_table(
        TableFunctionBuilder::new("ivf_hybrid_embed")
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .param(TypeId::Varchar)
            .named_param("top_k", TypeId::BigInt)
            .named_param("per_path", TypeId::BigInt)
            .named_param("nprobe", TypeId::BigInt)
            .named_param("k_rrf", TypeId::BigInt)
            .bind(embed_bind)
            .init(hybrid_init)
            .scan(hybrid_scan),
    )?;

    con.register_scalar(
        ScalarFunctionBuilder::new("ivf_version")
            .returns(TypeId::Varchar)
            .function(version_fn),
    )?;

    // ivf_levenshtein(a, b)            -> INTEGER, saturated at 3
    // ivf_levenshtein(a, b, maxd)      -> INTEGER, saturated at maxd
    // (Meilisearch-style typo tiers: exact=0, 1-typo, 2-typo; >2 = not a typo)
    con.register_scalar_set(
        ScalarFunctionSetBuilder::new("ivf_levenshtein")
            .overload(
                ScalarOverloadBuilder::new()
                    .param(TypeId::Varchar)
                    .param(TypeId::Varchar)
                    .returns(TypeId::Integer)
                    .function(lev_2_fn),
            )
            .overload(
                ScalarOverloadBuilder::new()
                    .param(TypeId::Varchar)
                    .param(TypeId::Varchar)
                    .param(TypeId::Integer)
                    .returns(TypeId::Integer)
                    .function(lev_3_fn),
            ),
    )?;

    Ok(())
}

quack_rs::entry_point_v2!(ivf_init_c_api, |con| register_all(con));
