/// A type alias for handling errors throughout meshopt
pub type Result<T> = std::result::Result<T, Error>;

/// An error that can occur
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An error that occurred interfacing with native code through FFI.
    #[error("native error: {0}")]
    Native(i32),

    /// An error that occurred while accessing or allocating memory
    #[error("memory error: {0}")]
    Memory(std::borrow::Cow<'static, str>),

    /// An error that occurred while parsing a data source
    #[error("parse error: {0}")]
    Parse(String),

    /// An error that occurred while working with a file path.
    #[error("path error: {0}")]
    Path(std::path::PathBuf),

    /// Generally, these errors correspond to bugs in this library.
    #[error("BUG: Please report this bug with a backtrace to https://github.com/gwihlidal/meshopt-rs\n{0}")]
    Bug(String),

    /// An error occurred while reading/writing a configuration
    #[error("config error: {0}")]
    Config(String),

    /// An unexpected I/O error occurred.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    // An error occurred while parsing a number in a free-form query.
    //Number,
}

impl Error {
    #[inline]
    pub(crate) fn memory(msg: &'static str) -> Self {
        Self::Memory(std::borrow::Cow::Borrowed(msg))
    }

    #[inline]
    pub(crate) fn memory_dynamic(msg: String) -> Self {
        Self::Memory(std::borrow::Cow::Owned(msg))
    }
}

#[inline]
pub(crate) fn error_or<T>(code: i32, ok: T) -> Result<T> {
    if code == 0 {
        Ok(ok)
    } else {
        Err(Error::Native(code))
    }
}
