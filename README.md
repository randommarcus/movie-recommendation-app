# 🎬 Movie Recommendation App

A content-based movie recommendation system built with Python and Streamlit. This app suggests films based on a comprehensive set of attributes, including genre, actors, directors, and plot synopsis from a Rotten Tomatoes dataset.

---

## 📜 Description

This project implements a content-based filtering algorithm to recommend movies. A user selects a movie from a dropdown list, and the application leverages natural language processing techniques to find and display 10 other movies with the most similar content profiles. The entire user interface is built with Streamlit, making it simple and interactive.

---

## ✨ Features

* **Interactive UI:** A clean and user-friendly web interface built with Streamlit.
* **Comprehensive Feature Set:** Recommendations are based on a rich combination of features, including name, description, genre, actors, directors, plot, and even ratings data.
* **Content-Based Logic:** Utilizes TF-IDF and Cosine Similarity to intelligently match movies based on their content.
* **Simple and Fast:** Quickly generates recommendations with a single button click.

---

## 🛠️ Tech Stack

* **Language:** `Python`
* **Libraries:**
    * `pandas`: For data manipulation and analysis.
    * `scikit-learn`: For machine learning (TF-IDF and Cosine Similarity).
    * `streamlit`: For building the interactive web app.

---

## ⚙️ How It Works

The recommendation engine follows a content-based filtering methodology:

1.  **Feature Engineering:** Key features for each movie (`name`, `genre`, `actors`, `directors`, `synopsis`, etc.) are combined into a single text string, or "content soup".
2.  **Vectorization:** The `TfidfVectorizer` from scikit-learn converts this collection of text strings into a numerical matrix. This matrix represents the importance of each word relative to each movie.
3.  **Similarity Calculation:** `Cosine Similarity` is used to calculate the similarity between all pairs of movie vectors. This results in a similarity matrix where a higher score indicates a greater content similarity.
4.  **Recommendation:** When a user selects a movie, the system looks up its similarity scores against all other movies, sorts them, and returns the top 10 matches.

---

## 🚀 Setup and Installation

To run this project locally, please follow these steps:

**1. Clone the repository:**

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

**2. Create a virtual environment (recommended):**

**For macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**3. Install the required dependencies:**
First, ensure you have a requirements.txt file by running pip freeze > requirements.txt in your activated environment.
```bash
pip install -r requirements.txt
```

**4. Download the dataset:**
You will need the rotten_tomatoes_movies_2025.csv dataset.
Place the CSV file in the root directory of the project.

**5. Run the Streamlit app:**
```bash
streamlit run app.py
```
Open your web browser and navigate to the local URL provided by Streamlit.

---

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

---

