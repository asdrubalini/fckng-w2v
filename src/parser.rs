use core::str;
use std::{collections::HashMap, fs::File, path::Path};

use memmap2::MmapOptions;
use nom::{
    bytes::complete::{tag, take, take_till1},
    character::complete::digit1,
    multi::count,
    number::complete::le_f32,
    IResult,
};

/// Parses an ASCII-encoded `u32` value from a byte slice, terminated by a specified ASCII character.
/// Fails if the number does not fit in an u32, if it is not terminated, or if
/// it is terminated by a different character.
fn ascii_u32_terminated_by(input: &[u8], terminator: u8) -> IResult<&[u8], u32> {
    // Parse the digit
    let (input, n) = digit1(input)?;

    // Consume the terminator
    let (input, _) = tag(&[terminator])(input)?;

    let n_str = str::from_utf8(n).unwrap(); // will always be a valid UTF-8 str as digit1 did not return an Err
    let n = n_str.parse::<u32>().unwrap(); // will always be a valid digit as digit1 did not return an Err

    Ok((input, n))
}

#[derive(Debug)]
pub(crate) struct Word2VecHeader {
    pub(crate) embeddings_count: u32,
    pub(crate) embeddings_dim: u32,
}

impl Word2VecHeader {
    pub(crate) fn parse(input: &[u8]) -> IResult<&[u8], Self> {
        // The header is encoded like this:
        // <ASCII embeddings_count><SPACE><ASCII embeddings_dim><LF>

        let (bytes, embeddings_count) = ascii_u32_terminated_by(input, b' ').unwrap();
        let (bytes, embeddings_dim) = ascii_u32_terminated_by(bytes, 0x0A).unwrap(); // 0x0A is a Line Feed

        let header = Word2VecHeader {
            embeddings_count,
            embeddings_dim,
        };

        Ok((bytes, header))
    }
}

#[derive(Debug)]
pub(crate) struct Word2VecEmbedding {
    pub(crate) word: String, // TODO: switch from String to &str using the lifetime of the mmap
    pub(crate) embedding: Vec<f32>,
}

fn string_terminated_by(input: &[u8], terminator: u8) -> IResult<&[u8], String> {
    // Take everything till terminator
    let (input, s) = take_till1(|c| c == terminator)(input)?;

    // Consume the terminator
    let (input, _) = tag(&[terminator])(input)?;

    // Turn the bytes into a String
    let s = String::from_utf8_lossy(s).to_string();

    Ok((input, s))
}

impl Word2VecEmbedding {
    pub(crate) fn parse(input: &[u8], embeddings_dim: u32) -> IResult<&[u8], Self> {
        // Each embedding is encoded like this:
        // <ASCII word><SPACE><N adjacent 32-bit floats with little endian ordering>

        let (bytes, word) = string_terminated_by(input, b' ').unwrap();

        // we have f32_len * embeddings_dim bytes that represents our embeddings
        let (bytes, embedding) = take(embeddings_dim as usize * std::mem::size_of::<f32>())(bytes)?;

        // dimensions are stored next to each other as 32-bit floats with little endian ordering
        let (remaning, embedding) = count(le_f32, 300usize)(embedding)?;

        assert_eq!(remaning.len(), 0); // we should be at the end of what we've taken

        Ok((bytes, Word2VecEmbedding { word, embedding }))
    }
}

pub(crate) struct Word2Vec {
    header: Word2VecHeader,
    embeddings: HashMap<String, Word2VecEmbedding>,
}

impl Word2Vec {
    pub(crate) fn new(file: impl AsRef<Path>) -> Self {
        let file = File::open(file).unwrap();
        // premature optimization
        // TODO: benchmark this vs. reading it normally
        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        let bytes = mmap.as_ref();

        let (bytes, header) = Word2VecHeader::parse(bytes).expect("Cannot parse file header.");

        let (bytes, embeddings_vec) = count(
            |b| Word2VecEmbedding::parse(b, header.embeddings_dim),
            header.embeddings_count as usize,
        )(bytes)
        .expect("Cannot parse embeddings.");

        assert_eq!(bytes.len(), 0); // we should be at the end of the file

        // turn the embeddings into an HashMap
        let embeddings = embeddings_vec
            .into_iter()
            .map(|e| (e.word.clone(), e))
            .collect();

        Word2Vec { header, embeddings }
    }

    /// Get the dictionary
    pub(crate) fn dictionary(&self) -> Vec<&str> {
        self.embeddings.keys().map(AsRef::as_ref).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_u32_terminated_by_ok() {
        let n = 923732897_u32;
        let delimiter = ';';

        let s = format!("{n}{delimiter}");

        let (input, n_parsed) = ascii_u32_terminated_by(s.as_bytes(), b';').unwrap();

        assert_eq!(input, &[]);
        assert_eq!(n_parsed, n);
    }

    #[test]
    #[should_panic]
    fn test_ascii_u32_terminated_by_wrong_digit() {
        let delimiter = ';';

        let s = format!("23374ciao123{delimiter}");

        ascii_u32_terminated_by(s.as_bytes(), b';').unwrap();
    }

    #[test]
    #[should_panic]
    fn test_ascii_u32_terminated_by_delimiter_not_found() {
        let delimiter = ';';

        let s = "236274.";

        ascii_u32_terminated_by(s.as_bytes(), delimiter as u8).unwrap();
    }
}
