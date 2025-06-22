# 🧠 GenAI Productivity Tracker

An intelligent daily planning assistant powered by Google Gemini that helps you organize, prioritize, and optimize your tasks using AI and machine learning.
---

## 📌 Project Overview

The **GenAI Productivity Tracker** is a Streamlit-based application that helps users:

- Plan their daily tasks using **LLM-driven scheduling**
- Track task completion with an interactive UI
- Visualize productivity trends using **Plotly**
- Predict future productivity using a **Random Forest model**
- Export personalized reports in **PDF** or text format
- Receive AI-generated reflections and habit suggestions based on daily performance

This project combines **AI, data science, and a beautiful UI** to improve how individuals manage their time and energy.

---

## 🚀 Features

- 📅 Plan Today – Use Gemini AI to generate personalized schedules based on your energy and task priority
- ✅ Task Completion – Mark tasks done and track your daily performance
- 📈 Analytics Dashboard – Visualize your progress and detect patterns
- 🔮 Predictive Insights – Forecast how likely you are to complete tasks based on past behavior
- 🔁 Task Recovery – Easily reschedule incomplete tasks
- 📤 Share Plans – Export plans as PDF or TXT
- 📄 Daily Reflection – Generate thoughtful summaries with productivity tips
- 🧠 Gemini AI Integration – Deep Work, Reflection, and Habit suggestions

---

## 🛠 Tech Stack

| Layer        | Tools                          |
|--------------|-------------------------------|
| 💬 AI Engine | Google Gemini 1.5 Flash        |
| 🌐 Frontend  | Streamlit                      |
| 📊 Charts    | Plotly                         |
| 🔍 ML Model  | Scikit-learn (Random Forest)   |
| 🧾 Reporting | FPDF (PDF generation)          |
| 💾 Storage   | JSON Files                     |
| ⚙️ Backend   | Python, dotenv                 |

---

## 📸 Screenshots

| Planner | Analytics | Reflection |
|--------|-----------|------------|
| ![](screenshots/plan_today.png) | ![](screenshots/analytics.png) | ![](screenshots/reflection.png) |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/genai-productivity-tracker.git
cd genai-productivity-tracker
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment

Create a `.env` file with your Gemini API key:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 📁 File Structure

```
genai-productivity-tracker/
│
├── app.py                 # Main application
├── task_history.json      # Daily plan records
├── task_completion.json   # Task completion data
├── requirements.txt       # Dependencies
├── .env                   # Gemini API key (not committed)
└── README.md              # Documentation
```

---

## 🧠 Machine Learning Model

A `RandomForestClassifier` is trained to predict whether a user will complete more than 70% of tasks on a given day, based on:

- Day of the week (weekday/weekend)
- Energy level (Low, Medium, High)
- Task count

It helps generate **predictive insights** and **feature importance** to improve planning.

---

## 📊 Gemini AI Prompts

### Task Planning Prompt
- Categorizes tasks (e.g., Health, Work, Learning)
- Estimates time duration
- Uses Deep Work blocks for focus time
- Aligns with energy level and priorities

### Daily Reflection Prompt
- Reflects on success, challenges, lessons
- Suggests improvements and encouragement

### Habit Prompt
- Recommends 1–2 keystone habits backed by behavioral science
- Format: `✨ Drink water before checking phone (Science: Implementation intention)`

---

## 📤 Export Options

- **TXT** — clean text plan output
- **PDF** — formatted, printable summary
- **Calendar Integration** — Coming Soon!

---

## 💡 Example Use Cases

- Students planning assignments and self-study
- Professionals managing meetings and deep work
- Creators organizing routines and learning
- Anyone seeking mindful productivity

---

## 🤝 Contributing

Want to improve this project?

1. Fork the repo
2. Create your branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -m "Add feature"`
4. Push to the branch: `git push origin feature/new-feature`
5. Open a pull request

---

## 👤 Author

**Krisha Goti**  
M.S. Applied Data Science, USC  
📧 [goti@usc.edu](mailto:goti@usc.edu)  
🌐 [linkedin.com/in/krishagoti](https://linkedin.com/in/krishagoti)

