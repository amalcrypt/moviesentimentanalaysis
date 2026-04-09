import "./globals.css";

export const metadata = {
  title: "SentimentAI — ML Sentiment Analysis",
  description:
    "Analyze movie review sentiment using state-of-the-art classical ML and transformer models. Built with Scikit-learn, HuggingFace DistilBERT, FastAPI & Next.js.",
  keywords: ["sentiment analysis", "machine learning", "NLP", "DistilBERT", "portfolio"],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
