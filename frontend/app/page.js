"use client";

import { useState, useEffect } from "react";
import styles from "./page.module.css";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const MODEL_DISPLAY = {
  logistic_regression: { label: "Logistic Regression", icon: "🔹", type: "Classical" },
  naive_bayes: { label: "Naive Bayes", icon: "🔸", type: "Classical" },
  linear_svm: { label: "Linear SVM", icon: "🔷", type: "Classical" },
  distilbert: { label: "DistilBERT", icon: "🤖", type: "Transformer" },
};

export default function Home() {
  const [activeTab, setActiveTab] = useState("analyze");
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function analyzeSentiment() {
    if (!text.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim(), model: "distilbert" }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `API error ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      analyzeSentiment();
    }
  }

  return (
    <div className={styles.container}>
      {/* ── Navbar ──────────────────────────────────────────────────────── */}
      <nav className={styles.navbar}>
        <div className={styles.logo}>
          <span className={styles.logoIcon}>🧠</span>
          SentimentAI
        </div>
        <div className={styles.navLinks}>
          <span
            style={{
              padding: "8px 14px",
              fontSize: "0.8rem",
              color: "var(--accent-green)",
              fontWeight: 600,
            }}
          >
            ● Transformer Online
          </span>
        </div>
      </nav>

      {/* ── Hero ───────────────────────────────────────────────────────── */}
      <section className={styles.hero}>
        <h1 className={styles.heroTitle}>Sentiment Analysis</h1>
        <p className={styles.heroSubtitle}>
          Analyze movie review sentiment using classical ML and transformer
          models trained on the IMDB dataset
        </p>
      </section>

      {/* ── Tab Bar ────────────────────────────────────────────────────── */}
      <div className={styles.tabBar}>
        {["analyze", "evaluation", "about"].map((tab) => (
          <button
            key={tab}
            className={`${styles.tab} ${activeTab === tab ? styles.tabActive : ""}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab === "analyze" && "⚡ Analyze"}
            {tab === "evaluation" && "📊 Evaluation"}
            {tab === "about" && "ℹ️ About"}
          </button>
        ))}
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          TAB: ANALYZE
          ══════════════════════════════════════════════════════════════════ */}
      {activeTab === "analyze" && (
        <div className={styles.analyzeGrid}>
          {/* Left — Input */}
          <div className={`${styles.inputPanel} glass`}>
            <div className={styles.sectionLabel}>💬 Enter Review</div>
            <textarea
              id="review-input"
              className={styles.textarea}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Type or paste a movie review here...\n\nExample: "This film was a masterpiece of storytelling with incredible performances from the entire cast."\n\nPress Ctrl+Enter to analyze`}
            />
            <div className={styles.controlRow}>
              <button
                id="analyze-btn"
                className={styles.analyzeBtn}
                onClick={analyzeSentiment}
                disabled={loading || !text.trim()}
                style={{ width: "100%" }}
              >
                {loading ? <span className={styles.spinner}></span> : "🔍 Analyze"}
              </button>
            </div>
          </div>

          {/* Right — Result */}
          <div className={`${styles.resultPanel} glass`}>
            <div className={styles.sectionLabel}>📋 Result</div>

            {!result && !error && !loading && (
              <div className={styles.resultPlaceholder}>
                <div className={styles.placeholderIcon}>🎬</div>
                <div className={styles.placeholderText}>
                  Enter a movie review and click{" "}
                  <strong style={{ color: "var(--accent-purple-light)" }}>Analyze</strong>{" "}
                  to see the sentiment prediction
                </div>
              </div>
            )}

            {loading && (
              <div className={styles.resultPlaceholder}>
                <span className={styles.spinner} style={{ width: 32, height: 32 }}></span>
                <div className={styles.placeholderText}>Analyzing sentiment...</div>
              </div>
            )}

            {error && <div className={styles.errorMsg}>⚠️ {error}</div>}

            {result && !loading && (
              <div className={`${styles.resultContent} animate-in`}>
                <div className={styles.resultStatsGrid}>
                  <div
                    className={`${styles.sentimentBadge} ${
                      result.sentiment === "positive"
                        ? styles.sentimentBadgePositive
                        : styles.sentimentBadgeNegative
                    }`}
                  >
                    <div className={styles.sentimentEmoji}>
                      {result.sentiment === "positive" ? "😊" : "😞"}
                    </div>
                    <div
                      className={`${styles.sentimentLabel} ${
                        result.sentiment === "positive" ? styles.positive : styles.negative
                      }`}
                    >
                      {result.sentiment}
                    </div>
                  </div>

                  <div className={styles.metricsCol}>
                    <div className={styles.metricCard}>
                      <div className={styles.metricValue}>
                        {(result.confidence * 100).toFixed(1)}%
                      </div>
                      <div className={styles.metricLabel}>Confidence</div>
                    </div>
                    <div className={styles.metricCard}>
                      <div className={styles.metricValue} style={{ fontSize: "1.1rem" }}>
                        {MODEL_DISPLAY[result.model]?.icon}{" "}
                        {MODEL_DISPLAY[result.model]?.label || result.model}
                      </div>
                      <div className={styles.metricLabel}>Model Used</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════
          TAB: EVALUATION
          ══════════════════════════════════════════════════════════════════ */}
      {activeTab === "evaluation" && (
        <EvaluationTab />
      )}

      {/* ══════════════════════════════════════════════════════════════════
          TAB: ABOUT
          ══════════════════════════════════════════════════════════════════ */}
      {activeTab === "about" && (
        <div className={styles.aboutSection}>
          <div className={`glass ${styles.aboutCard}`} style={{ marginBottom: 24 }}>
            <div className={styles.aboutCardTitle}>🏗️ Architecture</div>
            <p style={{ color: "var(--text-secondary)", lineHeight: 1.7 }}>
              This project demonstrates an{" "}
              <strong style={{ color: "var(--text-primary)" }}>
                end-to-end machine learning pipeline
              </strong>{" "}
              for binary sentiment analysis on the{" "}
              <strong style={{ color: "var(--accent-purple-light)" }}>
                IMDB Movie Reviews
              </strong>{" "}
              dataset (50,000 reviews). It covers data preprocessing, feature
              engineering, model training, evaluation, and deployment.
            </p>
          </div>

          <div className={`glass ${styles.aboutCard}`} style={{ marginBottom: 24 }}>
            <div className={styles.aboutCardTitle}>🤖 Transformer Model</div>
            <ul className={styles.aboutList}>
              <li>
                <strong>DistilBERT</strong> — Fine-tuned on IMDB
              </li>
              <li>Knowledge distillation of BERT-base</li>
              <li>97% of BERT&apos;s performance at 60% the size</li>
            </ul>
            <p className={styles.aboutNote}>
              Trained with HuggingFace Transformers, early stopping, and FP16
              mixed-precision on GPU.
            </p>
          </div>

          <div className={`glass ${styles.aboutCard}`}>
            <div className={styles.aboutCardTitle}>🛠️ Tech Stack</div>
            <div className={styles.techChips}>
              {[
                "Python",
                "Scikit-learn",
                "PyTorch",
                "HuggingFace",
                "FastAPI",
                "Next.js",
                "NLTK",
                "Matplotlib",
                "Vercel",
              ].map((tech) => (
                <span key={tech} className={styles.chip}>
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Footer ─────────────────────────────────────────────────────── */}
      <footer className={styles.footer}>
        Built with <span className={styles.footerHeart}>❤</span> using
        Scikit-learn, HuggingFace, FastAPI & Next.js
      </footer>
    </div>
  );
}


/* ═══════════════════════════════════════════════════════════════════════════
   EVALUATION TAB COMPONENT
   ═══════════════════════════════════════════════════════════════════════════ */
function EvaluationTab() {
  const [transResults, setTransResults] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/results/transformer`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setTransResults)
      .catch(() => {});
  }, []);

  return (
    <div className={styles.evalSection}>
      <div className={styles.evalSectionTitle}>📈 Model Performance</div>

      {/* Accuracy Cards */}
      <div className={styles.accuracyGrid}>
        {transResults && transResults.eval_accuracy && (
          <div className={`glass ${styles.accuracyCard}`}>
            <div className={styles.accuracyValue}>
              {(transResults.eval_accuracy * 100).toFixed(1)}%
            </div>
            <div className={styles.accuracyLabel}>DistilBERT</div>
            <div className={styles.accuracyType}>Transformer</div>
          </div>
        )}
      </div>
    </div>
  );
}


