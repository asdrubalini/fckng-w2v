use parser::Word2Vec;

mod parser;

fn main() {
    let w2v = Word2Vec::new("./GoogleNews-vectors-negative300.bin");
    // dbg!(w2v.dictionary());
}
