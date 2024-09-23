use scraper::{Html, Selector};

fn main() -> anyhow::Result<()> {
    let html = include_str!("../page.html");
    let html = Html::parse_document(&html);

    let s = Selector::parse("article.message").unwrap();

    let messages = html.select(&s);
    for elem in messages {
        let author = elem.attr("data-author").unwrap();
        let message_url = elem.attr("itemid").unwrap();
        println!("{:#?}", message_url);
    }

    Ok(())
}
