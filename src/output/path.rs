use errors::*;
use std::fs::DirBuilder;
use std::path::{Path, PathBuf};
use time;

/// ´OutputPath´ represents a common path, which all files written to disk share.
///
/// The `.with_extension()` method allows for easy change of file extension, to
/// differentiate between the outputs.
#[derive(Clone)]
pub struct OutputPath {
    path: PathBuf,
}

impl OutputPath {
    pub fn new<'a>(root: &'a Path, prefix: &str) -> OutputPath {
        let id = create_output_id(prefix);

        OutputPath { path: root.join(&id).join(&id) }
    }

    pub fn create(&self) -> Result<()> {
        // create directory containing all produced files
        create_output_dir(&self.path.parent().ok_or("Cannot create output directory")?)
    }

    // Returns path with given file extension.
    pub fn with_extension(&self, ext: &str) -> PathBuf {
        self.path.with_extension(ext)
    }
}


/// Returns an ID based on prefix, time, and version for simulation output
fn create_output_id(prefix: &str) -> String {
    // Need to introduce placeholder `.cbor`, since otherwise the patch version
    // number is chopped of later, when using `.with_extension()` method later.
    let v = ::VERSION.replace(".", "_");
    format!("{prefix}-{time}_v{version}",
            prefix = prefix,
            time = &time::now().strftime("%Y-%m-%d_%H%M%S").unwrap().to_string(),
            version = v)
}


/// Creates own ouput directory in output path using id.
fn create_output_dir(path: &Path) -> Result<()> {
    DirBuilder::new().create(&path)
        .chain_err(|| format!("Unable to create output directory '{}'", &path.display()))
}
