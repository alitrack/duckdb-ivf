//! Pure-Rust edit distance with adjacent-transposition support (OSA
//! Damerau-Levenshtein, banded, max-distance early exit).
//!
//! Exposed as the scalar function `ivf_levenshtein(a, b) -> INTEGER` so SQL
//! macros can generate typo-tolerant query expansion against the fts
//! dictionary (`fts_main_*.raw_dict`) — the missing piece of Meilisearch-style
//! typo tolerance on the fts + ivf stack.
//!
//! Distance semantics (Meilisearch-compatible): an adjacent transposition
//! ("sahcin" vs "sachin") costs 1, NOT 2 as in plain Levenshtein. This
//! matters because transpositions are the single most common human typo
//! class; with plain Levenshtein they would eat the entire 1-edit budget of
//! 4-5 letter words and never be corrected.

/// OSA Damerau-Levenshtein with an early-exit band: returns `maxd` (saturated)
/// as soon as no cell in the current row can beat it. Handles char-level
/// UTF-8. Symmetric. O(la*lb) worst case, effectively O(len*maxd) for short
/// words with small maxd.
///
/// Recurrence (rows over `b`, columns over `a`, D[i][j] = dist(a[..i], b[..j])):
///   insert    D[i-1][j]   + 1
///   delete    D[i][j-1]   + 1
///   replace   D[i-1][j-1] + cost
///   swap      D[i-2][j-2] + 1   (only when a[i-1] != b[j-1], a[i-2] == b[j-1],
///                                b[i-1]... == a[j-2], i.e. a[i-2..i] == b[j-2..j])
pub fn levenshtein_bounded(a: &str, b: &str, maxd: usize) -> u32 {
    let la = a.chars().count();
    let lb = b.chars().count();
    if lb == 0 {
        return la as u32;
    }
    if la == 0 {
        return lb as u32;
    }
    if la.abs_diff(lb) > maxd {
        return maxd as u32;
    }
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    // Rolling rows: `up` = row i-1, `up2` = row i-2 (needed by the swap term).
    // Both start as row 0 so the general recurrence works for i = 1 too
    // (the swap term is guarded by i > 1 and never reads up2 then).
    let row0: Vec<i32> = (0..=lb).map(|j| j as i32).collect();
    let mut up2 = row0.clone();
    let mut up = row0;
    for i in 1..=la {
        let mut cur = vec![0; lb + 1];
        cur[0] = i as i32;
        let mut row_min = cur[0];
        for j in 1..=lb {
            let cost = if av[i - 1] == bv[j - 1] { 0 } else { 1 };
            let mut v = (up[j] + 1).min(cur[j - 1] + 1).min(up[j - 1] + cost);
            if cost > 0 && i > 1 && j > 1 && av[i - 2] == bv[j - 1] && av[i - 1] == bv[j - 2] {
                v = v.min(up2[j - 2] + 1);
            }
            cur[j] = v;
            if cur[j] < row_min {
                row_min = cur[j];
            }
        }
        if row_min > maxd as i32 {
            return maxd as u32;
        }
        up2 = up;
        up = cur;
    }
    let d = up[lb];
    if d > maxd as i32 {
        maxd as u32
    } else {
        d as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        assert_eq!(levenshtein_bounded("kitten", "sitting", 10), 3);
        assert_eq!(levenshtein_bounded("", "", 10), 0);
        assert_eq!(levenshtein_bounded("abc", "", 10), 3);
        assert_eq!(levenshtein_bounded("", "abc", 10), 3);
        assert_eq!(levenshtein_bounded("oil", "oil", 10), 0);
        assert_eq!(levenshtein_bounded("cruide", "crude", 10), 1);
        assert_eq!(levenshtein_bounded("cruede", "crude", 10), 1);
        assert_eq!(levenshtein_bounded("oii", "oil", 10), 1);
        assert_eq!(levenshtein_bounded("hello", "world", 10), 4);
    }

    #[test]
    fn transposition_costs_one() {
        assert_eq!(levenshtein_bounded("sahcin", "sachin", 10), 1);
        assert_eq!(levenshtein_bounded("sachin", "sahcin", 10), 1);
        assert_eq!(levenshtein_bounded("ab", "ba", 10), 1);
        assert_eq!(levenshtein_bounded("ba", "ab", 10), 1);
        assert_eq!(levenshtein_bounded("cruad", "crude", 10), 2); // not a pure swap
        assert_eq!(levenshtein_bounded("abba", "baab", 10), 2); // two swaps (OSA: no shared-subseq double count)
    }

    #[test]
    fn band_saturates() {
        assert_eq!(levenshtein_bounded("kitten", "sitting", 2), 2); // real 3 -> saturates to 2
        assert_eq!(levenshtein_bounded("abcdef", "zzzzzz", 2), 2);
        assert_eq!(levenshtein_bounded("oil", "oii", 2), 1);
        assert_eq!(levenshtein_bounded("sahcin", "sachin", 1), 1); // swap fits 1-edit budget
        assert_eq!(levenshtein_bounded("sahcin", "sachin", 0), 0);
    }

    #[test]
    fn utf8() {
        assert_eq!(levenshtein_bounded("café", "cafe", 10), 1);
    }

    #[test]
    fn length_gap() {
        assert_eq!(levenshtein_bounded("a", "abcdef", 3), 3); // real 5 -> sat 3
        assert_eq!(levenshtein_bounded("abcdef", "a", 3), 3); // symmetric
        assert_eq!(levenshtein_bounded("ab", "abc", 1), 1);
        assert_eq!(levenshtein_bounded("abc", "ab", 1), 1);
        assert_eq!(levenshtein_bounded("x", "y", 2), 1);
    }
}
