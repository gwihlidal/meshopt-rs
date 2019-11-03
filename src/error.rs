#![allow(dead_code)]

use failure::{Backtrace, Context, Fail};
use std::fmt;
use std::path::{Path, PathBuf};
use std::result;

/// A type alias for handling errors throughout meshopt
pub type Result<T> = result::Result<T, Error>;

/// An error that can occur
#[derive(Debug)]
pub struct Error {
    ctx: Context<ErrorKind>,
}

impl Error {
    /// Return the kind of this error.
    pub fn kind(&self) -> &ErrorKind {
        self.ctx.get_context()
    }

    pub fn native(error_code: i32) -> Error {
        Error::from(ErrorKind::Native(error_code))
    }

    pub fn memory<T: AsRef<str>>(msg: T) -> Error {
        Error::from(ErrorKind::Memory(msg.as_ref().to_string()))
    }

    pub fn parse<T: AsRef<str>>(msg: T) -> Error {
        Error::from(ErrorKind::Parse(msg.as_ref().to_string()))
    }

    pub fn bug<T: AsRef<str>>(msg: T) -> Error {
        Error::from(ErrorKind::Bug(msg.as_ref().to_string()))
    }

    pub fn config<T: AsRef<str>>(msg: T) -> Error {
        Error::from(ErrorKind::Config(msg.as_ref().to_string()))
    }

    pub fn number<E: Fail>(err: E) -> Error {
        Error::from(err.context(ErrorKind::Number))
    }

    pub fn io<E: Fail>(err: E) -> Error {
        Error::from(err.context(ErrorKind::Io))
    }
}

impl Fail for Error {
    fn cause(&self) -> Option<&dyn Fail> {
        self.ctx.cause()
    }

    fn backtrace(&self) -> Option<&Backtrace> {
        self.ctx.backtrace()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.ctx.fmt(f)
    }
}

/// The specific kind of error that can occur.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    /// An error that occurred interfacing with native code through FFI.
    Native(i32),

    /// An error that occurred while accessing or allocating memory
    Memory(String),

    /// An error that occurred while parsing a data source
    Parse(String),

    /// An error that occurred while working with a file path.
    Path(PathBuf),

    /// Generally, these errors correspond to bugs in this library.
    Bug(String),

    /// An error occurred while reading/writing a configuration
    Config(String),

    /// An unexpected I/O error occurred.
    Io,

    /// An error occurred while parsing a number in a free-form query.
    Number,

    /// Hints that destructuring should not be exhaustive.
    ///
    /// This enum may grow additional variants, so this makes sure clients
    /// don't count on exhaustive matching. (Otherwise, adding a new variant
    /// could break existing code.)
    #[doc(hidden)]
    __Nonexhaustive,
}

impl ErrorKind {
    /// A convenience routine for creating an error associated with a path.
    pub(crate) fn path<P: AsRef<Path>>(path: P) -> ErrorKind {
        ErrorKind::Path(path.as_ref().to_path_buf())
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ErrorKind::Native(ref msg) => write!(f, "native error: {}", msg),
            ErrorKind::Memory(ref msg) => write!(f, "memory error: {}", msg),
            ErrorKind::Parse(ref msg) => write!(f, "parse error: {}", msg),
            ErrorKind::Path(ref path) => write!(f, "{}", path.display()),
            ErrorKind::Bug(ref msg) => {
                let report = "Please report this bug with a backtrace at \
                              https://github.com/gwihlidal/meshopt-rs";
                write!(f, "BUG: {}\n{}", msg, report)
            }
            ErrorKind::Config(ref msg) => write!(f, "config error: {}", msg),
            ErrorKind::Io => write!(f, "I/O error"),
            ErrorKind::Number => write!(f, "error parsing number"),
            ErrorKind::__Nonexhaustive => panic!("invalid error"),
        }
    }
}

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Error {
        Error::from(Context::new(kind))
    }
}

impl From<Context<ErrorKind>> for Error {
    fn from(ctx: Context<ErrorKind>) -> Error {
        Error { ctx }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::io(err)
        //Error::from(Context::new(ErrorKind::Io))
    }
}
