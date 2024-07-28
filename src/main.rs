#![allow(dead_code)]

use std::{fs::File, time::Instant};

use byte_unit::Byte;
use parser::Word2Vec;

mod parser;

fn main() {
    let f = "./GoogleNews-vectors-negative300.bin";

    let file_len: Byte = File::open(f).unwrap().metadata().unwrap().len().into();

    let start = Instant::now();
    let w2v = Word2Vec::new(f);
    let stop = Instant::now();

    let took = stop - start;
    let bytes_per_second = file_len.divide(took.as_secs() as usize).unwrap();

    println!(
        "{:.2} per second",
        bytes_per_second.get_appropriate_unit(byte_unit::UnitType::Binary)
    );
}
