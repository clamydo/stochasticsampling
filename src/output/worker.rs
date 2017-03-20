use super::path::OutputPath;
use bincode;
use errors::*;
use serde_cbor;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::mem::transmute;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use stochasticsampling::simulation::Snapshot;
use stochasticsampling::simulation::output::Output;
use stochasticsampling::simulation::settings::{OutputFormat, Settings};


/// Message type for the IO worker thread channel.
// TODO find out, if settings can be taken by reference
pub enum IOWorkerMsg {
    Quit,
    Snapshot(Snapshot),
    Output(Output),
    Settings(Settings),
}

struct OutputFile {
    path: PathBuf,
    format: OutputFormat,
}

pub struct Worker {
    tx: SyncSender<IOWorkerMsg>,
    io_worker: JoinHandle<Result<()>>,
    output_file: OutputFile,
}

impl Worker {
    pub fn new(io_queue_size: usize,
               output_path: &OutputPath,
               output_format: OutputFormat)
               -> Result<Worker> {

        // Create communication channel for thread
        let (tx, rx) = mpsc::sync_channel::<IOWorkerMsg>(io_queue_size);

        let (file, output_file) = prepare_output_file(output_path, output_format)
            .chain_err(|| "Cannot create output file.")?;


        // clone, so it can be moved into thread closure
        let op = output_path.clone();
        let of = output_file.format;

        // Spawn worker thread, that periodically flushes collections of simulation
        // states to disk.
        let io_worker = thread::spawn(move || dispatch(rx, file, of, op));

        Ok(Worker {
               io_worker: io_worker,
               tx: tx,
               output_file: output_file,
           })
    }

    pub fn write_metadata(&self, settings: Settings) -> Result<()> {
        self.tx.send(IOWorkerMsg::Settings(settings)).chain_err(||
            "Cannot write metadata to output file."
        )
    }

    pub fn append(&self, output: Output) -> Result<()> {
        self.tx
            .send(IOWorkerMsg::Output(output))
            .chain_err(|| "Cannot append data to file.")?;

        Ok(())
    }

    pub fn write_snapshot(&self, snapshot: Snapshot) -> Result<()> {
        self.tx
            .send(IOWorkerMsg::Snapshot(snapshot))
            .chain_err(|| "Cannot write snapshot.")?;

        Ok(())
    }

    pub fn get_output_filepath(&self) -> &Path {
        &self.output_file.path
    }

    pub fn quit(self) -> Result<()> {
        self.tx.send(IOWorkerMsg::Quit).unwrap();

        match self.io_worker.join() {
            Ok(v) => v,
            Err(e) => panic!("Cannot join worker thread, {:?}", e),
        }
    }
}


/// Creates an output file. Already writes header for metadata.
fn prepare_output_file(path: &OutputPath, format: OutputFormat) -> Result<(File, OutputFile)> {
    let fileext = match format {
        OutputFormat::CBOR => "cbor",
        OutputFormat::Bincode => "bincode",
    };

    let filepath = path.with_extension(fileext);

    let file = File::create(&filepath)
        .chain_err(|| format!("couldn't create output file '{}'.", filepath.display()))?;

    let ofile = OutputFile {
        path: filepath,
        format: format,
    };

    Ok((file, ofile))
}


fn dispatch(rx: Receiver<IOWorkerMsg>,
            mut file: File,
            format: OutputFormat,
            path: OutputPath)
            -> Result<()> {

    let mut snapshot_counter = 0;

    let index_file_path = path.with_extension("index");
    let mut index_file =
        File::create(&index_file_path).chain_err(|| {
                           format!("Cannot create index file '{}'",
                                   index_file_path.display())
                       })?;


    loop {
        match rx.recv().unwrap() {
            IOWorkerMsg::Quit => break,

            IOWorkerMsg::Snapshot(s) => {
                snapshot_counter += 1;
                let filepath = path.with_extension(&format!("bincode.{}", snapshot_counter));

                let mut snapshot_file =
                    File::create(&filepath).chain_err(|| {
                                       format!("Cannot create snapshot file '{}'.",
                                               filepath.display())
                                   })?;

                bincode::serialize_into(&mut snapshot_file,
                                                    &s,
                                                    bincode::SizeLimit::Infinite).chain_err(|| {
                                                   format!("Cannot write snapshot with number {}",
                                                           snapshot_counter)
                                               })?
            }

            IOWorkerMsg::Output(v) => {
                // write starting offset of blob into index file
                let pos = file.seek(SeekFrom::Current(0)).unwrap();
                let pos_le: [u8; 8] = unsafe { transmute(pos.to_le()) };
                index_file.write_all(&pos_le).chain_err(|| "Failed to write into index file.")?;

                // write all snapshots into one cbor file
                match format {
                    OutputFormat::CBOR => {
                        serde_cbor::ser::to_writer_sd(&mut file, &v)
                        .chain_err(||
                            "Cannot write simulation output (format: CBOR).")?
                    }
                    OutputFormat::Bincode => {
                        bincode::serialize_into(
                            &mut file, &v,
                            bincode::SizeLimit::Infinite)
                        .chain_err(||
                            "Cannot write simulation output (format: bincode).")?
                    }
                }
            }

            IOWorkerMsg::Settings(v) => {
                // Serialize settings as first object in file
                match format {
                    OutputFormat::CBOR => serde_cbor::ser::to_writer_sd(&mut file, &v).unwrap(),
                    OutputFormat::Bincode => {
                        bincode::serialize_into(&mut file, &v, bincode::SizeLimit::Infinite)
                            .unwrap()
                    }
                }
            }
        }
    }

    Ok(())
}
