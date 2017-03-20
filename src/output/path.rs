use errors::*;
use std::fs::DirBuilder;
use std::path::{Path, PathBuf};
use time;

/// OutputPath represents a common path, which all files written to disk share.
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

        // create directory containing all produced files
        let dir = create_output_dir(&id, &root).unwrap();

        OutputPath { path: dir.join(&id) }
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
fn create_output_dir(id: &str, root_path: &Path) -> Result<PathBuf> {
    let dir = root_path.join(Path::new(id));
    DirBuilder::new().create(&dir)
        .chain_err(|| format!("Unable to create output directory '{}'", &dir.display()))?;

    Ok(dir)
}
