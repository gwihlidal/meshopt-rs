use crate::{Error, Result, ffi, validate_triangles};

/// Converts a previously vertex cache optimized triangle list to triangle
/// strip, stitching strips using restart index.
///
/// For maximum efficiency the index buffer being converted has to be
/// optimized for vertex cache first.
///
/// The `restart_index` should be 0xffff or 0xffffffff depending on index size,
/// or 0 to use degenerate triangles.
pub fn stripify(indices: &[u32], vertex_count: usize, restart_index: u32) -> Result<Vec<u32>> {
    validate_triangles(indices, vertex_count)?;
    let bound = unsafe { ffi::meshopt_stripifyBound(indices.len()) };
    let mut result: Vec<u32> = vec![0; bound];
    let index_count = unsafe {
        ffi::meshopt_stripify(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            vertex_count,
            restart_index,
        )
    };
    if index_count <= result.len() {
        result.truncate(index_count);
        result.shrink_to_fit();
        Ok(result)
    } else {
        Err(Error::memory("index count is larger than result"))
    }
}

/// Converts a triangle strip to a triangle list
pub fn unstripify(indices: &[u32], restart_index: u32) -> Result<Vec<u32>> {
    // a strip with fewer than 3 indices contains no triangles
    if indices.len() < 3 {
        return Ok(Vec::new());
    }
    let bound = unsafe { ffi::meshopt_unstripifyBound(indices.len()) };
    let mut result: Vec<u32> = vec![0; bound];
    let index_count = unsafe {
        ffi::meshopt_unstripify(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            restart_index,
        )
    };
    if index_count <= result.len() {
        result.truncate(index_count);
        result.shrink_to_fit();
        Ok(result)
    } else {
        Err(Error::memory("index count is larger than result"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stripify_disjoint_triangles() {
        // disjoint triangles are the worst case: with degenerate triangles each one after
        // the first takes 5 indices, with restart indices it takes 4
        let indices = [0, 1, 2, 3, 4, 5, 6, 7, 8];

        let strip = stripify(&indices, 9, 0).unwrap();
        assert_eq!(strip.len(), 13);

        let strip = stripify(&indices, 9, u32::MAX).unwrap();
        assert_eq!(strip.len(), 11);
        assert_eq!(unstripify(&strip, u32::MAX).unwrap(), indices);
    }

    #[test]
    fn test_stripify_invalid_input() {
        assert!(stripify(&[0, 1, 2, 3], 4, 0).is_err());
        assert!(stripify(&[0, 1, 3], 3, 0).is_err());
    }

    #[test]
    fn test_unstripify_short_input() {
        for len in 0..3 {
            let strip = vec![0; len];
            assert!(unstripify(&strip, 0).unwrap().is_empty());
        }
    }
}
