"""Generate word clouds for positive, negative, and neutral comments across all course CSVs."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

logger = logging.getLogger(__name__)

INPUT_DIR = Path("output/full_comments")
OUTPUT_DIR = Path("output")
FONT_PATH = "/usr/share/fonts/truetype/Gargi/Gargi.ttf"
WORDCLOUD_WIDTH = 1600
WORDCLOUD_HEIGHT = 800
STOPWORDS_EXTRA = {"course", "hai", "please", "also", "really", "much", "ni", "ka", "ke", "ki", "se", "na"}

SENTIMENT_CONFIG: dict[str, dict] = {
    "positive": {"colormap": "tab20",   "title": "Positive Comments — Word Cloud"},
    "negative": {"colormap": "Reds",    "title": "Negative Comments — Word Cloud"},
    "neutral":  {"colormap": "Blues",   "title": "Neutral Comments — Word Cloud"},
}


def _load_comments_by_sentiment(input_dir: Path, sentiment: str) -> str:
    """Load and concatenate all comments of a given sentiment from CSV files.

    Args:
        input_dir: Directory containing per-course CSV files.
        sentiment: Sentiment label to filter on (positive, negative, neutral).

    Returns:
        Single string of all matching comments joined together.
    """
    all_comments: list[str] = []
    for csv_file in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(csv_file)
        filtered = df.loc[
            df["predicted sentiment"].str.lower() == sentiment, "comment"
        ]
        all_comments.extend(filtered.dropna().tolist())

    logger.info("Loaded %d '%s' comments from %s", len(all_comments), sentiment, input_dir)
    return " ".join(all_comments)


def _generate_wordcloud(text: str, output_path: Path, title: str, colormap: str) -> None:
    """Generate a word cloud image from text and save to disk.

    Args:
        text: Combined text to generate the word cloud from.
        output_path: File path to save the resulting PNG image.
        title: Title displayed above the word cloud.
        colormap: Matplotlib colormap name to use for the word cloud.
    """
    wc = WordCloud(
        width=WORDCLOUD_WIDTH,
        height=WORDCLOUD_HEIGHT,
        background_color="white",
        colormap=colormap,
        font_path=FONT_PATH,
        stopwords=WordCloud().stopwords | STOPWORDS_EXTRA,
        max_words=150,
        min_word_length=2,
        collocations=False,
    )
    wc.generate(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close()

    logger.info("Word cloud saved to %s", output_path)


def main() -> None:
    """Entry point: generate word clouds for all three sentiment categories."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    for sentiment, config in SENTIMENT_CONFIG.items():
        text = _load_comments_by_sentiment(INPUT_DIR, sentiment)
        if not text.strip():
            logger.warning("No '%s' comments found — skipping.", sentiment)
            continue

        output_path = OUTPUT_DIR / f"{sentiment}_wordcloud.png"
        _generate_wordcloud(text, output_path, config["title"], config["colormap"])

    logger.info("Done. Word clouds saved in %s/", OUTPUT_DIR)


if __name__ == "__main__":
    main()
