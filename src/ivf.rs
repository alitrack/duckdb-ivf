//! Pure-Rust IVF-Flat approximate nearest-neighbor index.
//!
//! No external dependencies (deterministic k-means over a SplitMix64 PRNG), so the
//! extension builds fully offline. This is the self-contained core; the quack-rs
//! binding layer (`lib.rs`) moves data in and out via the C-API and keeps built
//! indexes in a process-global registry keyed by name.
//!
//! IVF-Flat ("inverted file, flat vectors"):
//!   * train: k-means places `nlist` centroids in the vector space; every stored
//!     vector is assigned to its nearest centroid and appended to that list.
//!   * search: the query is matched against all centroids; the `nprobe` nearest
//!     centroids' lists are scanned exactly, and the top-k are returned.
//!
//! With `nprobe == nlist` the search is exhaustive (recall 1.0); smaller `nprobe`
//! trades recall for speed. Distances are inner product over (optionally
//! L2-normalized) vectors, i.e. cosine similarity — the same convention the
//! evaluation harness (bge-m3, L2-normalized) uses.

use std::collections::HashMap;
use std::sync::Mutex;

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic RNG (SplitMix64) — no `rand` dependency, offline-buildable.
// ─────────────────────────────────────────────────────────────────────────────

struct SplitMix64(u64);
impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform float in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0f32 / 16_777_216.0)
    }
    /// Uniform integer in [0, n).
    fn next_u32_range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IVF index
// ─────────────────────────────────────────────────────────────────────────────

/// A trained IVF-Flat index.
pub struct Ivf {
    pub dim: usize,
    pub n: usize,
    pub nlist: usize,
    /// Centroids, row-major `[nlist * dim]`.
    pub centroids: Vec<f32>,
    /// For each vector, the centroid (list) it was assigned to.
    list_of: Vec<usize>,
    /// Per-list vector ids (row-major order preserved within a list).
    pub lists: Vec<Vec<usize>>,
    /// All stored vectors, row-major `[n * dim]`, addressed by dense index.
    pub data: Vec<f32>,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

impl Ivf {
    /// Train an IVF index from `n` row-major vectors of dimension `dim`.
    ///
    /// * `nlist` is clamped to `[1, n]`; `niter` is the k-means iteration cap.
    /// * `normalize=true` L2-normalizes every vector (cosine/inner-product search);
    ///   when false, vectors are used as-is (plain inner product).
    pub fn train(
        data: &[f32],
        dim: usize,
        nlist: usize,
        niter: usize,
        normalize: bool,
        seed: u64,
    ) -> Result<Self, String> {
        if dim == 0 {
            return Err("ivf: dim must be > 0".into());
        }
        let n = data.len() / dim;
        if data.len() != n * dim {
            return Err(format!("ivf: data len {} is not a multiple of dim {}", data.len(), dim));
        }
        if n == 0 {
            return Err("ivf: no vectors to index".into());
        }
        let nlist = nlist.clamp(1, n);

        let mut data = data.to_vec();
        if normalize {
            for row in data.chunks_exact_mut(dim) {
                l2_normalize(row);
            }
        }

        // k-means: init centroids by spread sampling, then iterate.
        let mut rng = SplitMix64::new(seed);
        let mut centroids = vec![0.0f32; nlist * dim];
        for c in 0..nlist {
            let pick = (c as f64 * n as f64 / nlist as f64) as usize;
            let pick = pick % n;
            centroids[c * dim..(c + 1) * dim].copy_from_slice(&data[pick * dim..(pick + 1) * dim]);
        }

        let mut assignment = vec![0usize; n];
        for _it in 0..niter.max(1) {
            let mut counts = vec![0usize; nlist];
            let mut sums: Vec<f32> = vec![0.0; nlist * dim];
            for i in 0..n {
                let row = &data[i * dim..(i + 1) * dim];
                let mut best = 0usize;
                let mut best_s = f32::NEG_INFINITY;
                for c in 0..nlist {
                    let s = dot(row, &centroids[c * dim..(c + 1) * dim]);
                    if s > best_s {
                        best_s = s;
                        best = c;
                    }
                }
                assignment[i] = best;
                counts[best] += 1;
                let dst = &mut sums[best * dim..(best + 1) * dim];
                for (d, x) in dst.iter_mut().zip(row.iter()) {
                    *d += x;
                }
            }
            // Update non-empty centroids; re-seed empty ones (deterministic pick).
            for c in 0..nlist {
                if counts[c] > 0 {
                    for (d, x) in centroids[c * dim..(c + 1) * dim].iter_mut().enumerate() {
                        *x = sums[c * dim + d] / counts[c] as f32;
                    }
                    if normalize {
                        l2_normalize(&mut centroids[c * dim..(c + 1) * dim]);
                    }
                } else {
                    let pick = rng.next_u32_range(n);
                    centroids[c * dim..(c + 1) * dim].copy_from_slice(&data[pick * dim..(pick + 1) * dim]);
                }
            }
        }

        // Build inverted lists from the final assignment.
        let mut lists: Vec<Vec<usize>> = vec![Vec::new(); nlist];
        for i in 0..n {
            lists[assignment[i]].push(i);
        }

        Ok(Ivf {
            dim,
            n,
            nlist,
            centroids,
            list_of: assignment,
            lists,
            data,
        })
    }

    /// Exact inner-product search over all stored vectors (recall reference /
    /// also usable as the brute force when nlist is 1).
    pub fn search_exact(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = (0..self.n)
            .map(|i| (i, dot(query, &self.data[i * self.dim..(i + 1) * self.dim])))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Approximate search: probe the `nprobe` nearest centroids.
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<(usize, f32)> {
        let nprobe = nprobe.clamp(1, self.nlist);
        // Nearest centroids by inner product.
        let mut cents: Vec<(usize, f32)> = (0..self.nlist)
            .map(|c| (c, dot(query, &self.centroids[c * self.dim..(c + 1) * self.dim])))
            .collect();
        cents.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        cents.truncate(nprobe);

        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(k);
        for (c, _) in cents {
            for &i in self.lists[c].iter() {
                scored.push((i, dot(query, &self.data[i * self.dim..(i + 1) * self.dim])));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}
