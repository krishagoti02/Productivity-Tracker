import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
import plotly.express as px
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fpdf import FPDF


def create_pdf_report(content, title="Productivity Report"):
    import re
    
    # Function to remove emojis and non-ASCII characters
    def clean_text(text):
        # Remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Keep only ASCII characters
        return text.encode('ascii', 'ignore').decode('ascii')
    
    pdf = FPDF()
    pdf.add_page()
    
    # Use built-in font (no need for external files)
    pdf.set_font("Arial", 'B', 16)
    
    # Header with cleaned text
    pdf.cell(0, 10, clean_text(title), 0, 1, 'C')
    pdf.ln(10)
    
    # Body
    pdf.set_font("Arial", size=12)
    for line in content.split('\n'):
        clean_line = clean_text(line)
        # Handle markdown headings
        if clean_line.startswith('###'):
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, clean_line.replace('###', '').strip(), 0, 1)
            pdf.set_font('Arial', '', 12)
        else:
            pdf.multi_cell(0, 10, clean_line)
    
    return pdf.output(dest='S').encode('latin1')

# --- Setup ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

HISTORY_FILE = "task_history.json"
COMPLETION_FILE = "task_completion.json"

# --- Data Loading Helpers ---
def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# --- Predictive Analytics Functions ---
def prepare_analytics_data():
    """Prepares data for predictive modeling with improved data validation"""
    completion_data = load_json(COMPLETION_FILE)
    history_data = load_json(HISTORY_FILE)
    
    records = []
    for date, entries in completion_data.items():
        # Skip dates with no tasks or missing history
        if not entries.get("all") or date not in history_data:
            continue
            
        try:
            weekday = datetime.strptime(date, "%Y-%m-%d").weekday()
            records.append({
                "date": date,
                "weekday": weekday,
                "energy": history_data[date].get("energy", "Medium"),
                "total_tasks": len(entries["all"]),
                "completed": len(entries.get("completed", [])),
                "completion_rate": len(entries.get("completed", [])) / len(entries["all"]) if len(entries["all"]) > 0 else 0,
            })
        except (KeyError, ValueError) as e:
            print(f"Skipping malformed data for {date}: {str(e)}")
            continue
            
    return pd.DataFrame(records)

def train_model():
    """Trains prediction model with enhanced error handling"""
    df = prepare_analytics_data()
    
    # Check for sufficient and valid data
    if len(df) < 10:  # Increased minimum data points for better accuracy
        #st.warning("Need at least 10 days of completed tasks for reliable predictions")
        return None, None, None
    
    # Feature engineering
    try:
        df["is_weekend"] = df["weekday"].apply(lambda x: x >= 5)
        energy_map = {"Low": 0, "Medium": 1, "High": 2}
        df["energy_encoded"] = df["energy"].map(energy_map)
        
        # Ensure we have variation in features
        if df["energy_encoded"].nunique() < 2 or df["is_weekend"].nunique() < 2:
            st.warning("Not enough variation in energy levels or weekdays/weekends")
            return None, None, None

        X = df[["weekday", "is_weekend", "energy_encoded", "total_tasks"]]
        y = (df["completion_rate"] > df["completion_rate"].median()).astype(int)  # More robust threshold
        
        # Check class balance
        if y.value_counts().min() < 3:  # At least 3 samples in minority class
            st.warning("Not enough examples of both completion outcomes")
            return None, None, None

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        
        # Train model with balanced class weights
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        return model, X_test, y_test
        
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, None, None


# --- Cleanup Redundant Entries ---
completion_data = load_json(COMPLETION_FILE)
history_data = load_json(HISTORY_FILE)

for date in list(history_data.keys()):
    if date in completion_data:
        completed_set = set(completion_data[date].get("completed", []))
        input_set = set(history_data[date].get("input", []))
        if completed_set == input_set:
            del history_data[date]

save_json(history_data, HISTORY_FILE)

# --- UI Config ---
st.set_page_config(
    page_title="GenAI Productivity Tracker", 
    layout="wide",
    page_icon="🧠"
)

# --- Custom Styles ---
st.markdown("""
    <style>
    :root {
        --primary: #6e48aa;
        --secondary: #9d50bb;
        --accent: #4776e6;
        --light: #f8f9fa;
        --dark: #212529;
        --success: #28a745;
        --warning: #ffc107;
        --danger: #dc3545;
    }
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #fbfbfb;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        font-size: 16px;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        border-radius: 12px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .block-container {
        padding: 2rem;
        max-width: 1200px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #6e48aa 0%, #9d50bb 100%);
        color: white;
    }
    .sidebar .sidebar-content .stRadio label {
        color: white !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--primary);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 16px !important;
        background: white !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Main Header ---
st.markdown("""
<div style='text-align:center; margin-bottom:30px;'>
    <h1 style='color:#6e48aa; margin-bottom:0;'> GenAI Productivity Tracker</h1>
    <p style='color:#6e48aa; opacity:0.8;'>Your AI-powered daily planning companion</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom:30px;'>
        <h2 style='color:#6e48aa;'>📌 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    section = st.radio("", [
        "📅 Plan Today", "📂 Past Plans", "✅ Task Completion",
        "📈 Analytics", "🔁 Task Recovery", "📤 Share Plan",
        "📄 Daily Reflection"
    ], label_visibility="collapsed")

# --- Plan Today ---
if section == "📅 Plan Today":
    st.markdown("### ✍️ Plan Your Day")
    st.info("Let Gemini help you organize, prioritize, and schedule your tasks effectively.", icon="🤖")
    
    with st.expander("⚙️ Plan Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.date_input("📆 Date", datetime.now().date())
        with col2:
            energy = st.select_slider("⚡ Energy Level", ["Low", "Medium", "High"])
    
    st.markdown("### 📝 Your Tasks")
    tasks_input = st.text_area(
        "Enter one task per line:", 
        height=200,
        placeholder="Example tasks:\nComplete project proposal\nGym workout\nCall mom\nRead 30 pages\nPrepare presentation",
        help="Write each task on a new line. You'll prioritize them next."
    )
    
    if tasks_input.strip():
        task_list = [t.strip() for t in tasks_input.split("\n") if t.strip()]
        
        st.markdown("### 🏷️ Prioritize Tasks")
        tab1, tab2, tab3 = st.tabs(["🔴 High Priority", "🟡 Medium Priority", "⚪ Low Priority"])
        
        with tab1:
            st.markdown("**Critical tasks that need your focus**")
            high = st.multiselect("High priority tasks", task_list, key="high", label_visibility="collapsed")
        
        with tab2:
            st.markdown("**Important but not urgent tasks**")
            medium = st.multiselect("Medium priority tasks", [t for t in task_list if t not in high], key="medium", label_visibility="collapsed")
        
        with tab3:
            st.markdown("**Nice-to-have or routine tasks**")
            low = st.multiselect("Low priority tasks", [t for t in task_list if t not in high + medium], key="low", label_visibility="collapsed")
        
        unmarked = [t for t in task_list if t not in high + medium + low]
        
        if unmarked:
            st.warning(f"You have {len(unmarked)} unprioritized tasks. Consider assigning them a priority level.")
        
        if st.button("🚀 Generate Smart Plan", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing your tasks and creating optimal schedule..."):
                prompt = f"""
You are an expert productivity assistant creating a daily plan. Follow these instructions carefully:

**User Context:**
- Date: {selected_date}
- Energy Level: {energy}
- Total Tasks: {len(task_list)}
- High Priority: {len(high)}
- Medium Priority: {len(medium)}
- Low Priority: {len(low)}

**Tasks:**
{chr(10).join([f"• {t}" for t in task_list])}

**Create a comprehensive plan with:**
1. **Task Categorization** (Work, Health, Personal, Learning, etc.)
2. **Time Estimates** (15, 30, 45, 60 mins)
3. **Optimal Scheduling** based on energy levels
4. **Priority Alignment** with user's manual tags
5. **Buffer Times** between tasks

**Output Format:**
### 📅 Daily Plan for {selected_date}
**Energy Level:** {energy}

#### 🗂️ Task Overview
| Task | Category | Priority | Time Estimate |
|------|----------|----------|---------------|

#### ⏳ Time-Blocked Schedule
**Morning (High Energy)**
- 8:00-8:30: Task 1 [Category] (Priority: 🔴)
- 8:30-9:00: Task 2 [Category] (Priority: 🟡)

**Afternoon**
- [Continue with optimal schedule...]

#### 💡 Productivity Tips
[Include 2-3 personalized tips based on the plan]
"""
                try:
                    response = model.generate_content(prompt)
                    output = response.text
                    
                    st.markdown("### 📋 Your AI-Generated Plan")
                    st.markdown(output, unsafe_allow_html=True)
                    
                    # Save to history
                    history = load_json(HISTORY_FILE)
                    history[str(selected_date)] = {
                        "date": str(selected_date),
                        "energy": energy,
                        "input": task_list,
                        "manual_priorities": {"high": high, "medium": medium, "low": low},
                        "output": output
                    }
                    save_json(history, HISTORY_FILE)
                    
                    st.success("Plan saved successfully!", icon="✅")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error generating plan: {str(e)}", icon="❌")

# --- Past Plans ---
elif section == "📂 Past Plans":
    st.markdown("### 📂 Your Planning History")
    history = load_json(HISTORY_FILE)
    
    if not history:
        st.info("You haven't created any plans yet. Start by planning your day!")
    else:
        col1, col2 = st.columns([3,1])
        with col1:
            selected = st.selectbox(
                "Select a date", 
                sorted(history.keys(), reverse=True),
                format_func=lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%A, %B %d, %Y")
            )
        with col2:
            st.markdown("##")
            if st.button("🗑️ Delete Plan", type="secondary"):
                del history[selected]
                save_json(history, HISTORY_FILE)
                st.success("Plan deleted!")
                st.rerun()
        
        plan = history[selected]
        
        with st.expander(f"🔍 View Plan Details", expanded=True):
            st.markdown(f"""
            **📅 Date:** {selected}  
            **⚡ Energy Level:** {plan['energy']}  
            **📝 Total Tasks:** {len(plan['input'])}
            """)
            
            st.markdown("#### Original Tasks")
            st.write(plan['input'])
            
            st.markdown("#### AI-Generated Plan")
            st.markdown(plan['output'], unsafe_allow_html=True)
        
        # Completion status if available
        completion = load_json(COMPLETION_FILE)
        if selected in completion:
            completed = len(completion[selected]["completed"])
            total = len(completion[selected]["all"])
            st.metric("Completion Rate", f"{completed}/{total} tasks", f"{round(completed/total*100)}%")

# --- Task Completion ---
elif section == "✅ Task Completion":
    st.markdown("### ✅ Track Your Progress")
    history = load_json(HISTORY_FILE)
    completion = load_json(COMPLETION_FILE)
    
    if not history:
        st.info("No plans available. Create a plan first.")
    else:
        selected = st.selectbox(
            "Select date", 
            sorted(history.keys(), reverse=True),
            format_func=lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%A, %B %d, %Y")
        )
        
        tasks = history[selected]["input"]
        completed_tasks = completion.get(selected, {}).get("completed", [])
        
        st.markdown(f"#### Mark completed tasks for {selected}")
        
        # Improved task completion interface
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**All Tasks**")
            for i, task in enumerate(tasks):
                is_checked = task in completed_tasks
                if st.checkbox(task, value=is_checked, key=f"task_{i}"):
                    if task not in completed_tasks:
                        completed_tasks.append(task)
                else:
                    if task in completed_tasks:
                        completed_tasks.remove(task)
        
        with cols[1]:
            st.markdown("**Completion Summary**")
            if completed_tasks:
                progress = len(completed_tasks)/len(tasks)
                st.progress(progress)
                st.metric("Completed", f"{len(completed_tasks)}/{len(tasks)}", f"{round(progress*100)}%")
                
                st.markdown("**✅ Completed Tasks**")
                for task in completed_tasks:
                    st.markdown(f"- {task}")
            else:
                st.info("No tasks completed yet")
        
        # Save button
        if st.button("💾 Save Completion Status", type="primary"):
            completion[selected] = {
                "completed": completed_tasks,
                "all": tasks,
                "timestamp": str(datetime.now())
            }
            save_json(completion, COMPLETION_FILE)
            st.success("Completion status saved!")

# --- Analytics ---
elif section == "📈 Analytics":
    st.markdown("## 📊 Your Productivity Analytics")
    completion = load_json(COMPLETION_FILE)
    
    if not completion:
        st.info("✨ No completion data yet. Complete some tasks to see analytics!")
    else:
        # Date range selector with improved defaults
        dates = sorted([datetime.strptime(d, "%Y-%m-%d") for d in completion.keys()])
        min_date, max_date = dates[0], dates[-1]
        
        with st.expander("🗓️ Date Range Settings", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start date",
                    min_date,
                    min_value=min_date,
                    max_value=max_date
                )
            with col2:
                end_date = st.date_input(
                    "End date",
                    max_date,
                    min_value=min_date,
                    max_value=max_date
                )
        
        # Filter data with validation
        filtered_data = {
            d: v for d, v in completion.items() 
            if start_date <= datetime.strptime(d, "%Y-%m-%d").date() <= end_date
        }
        
        if not filtered_data:
            st.warning("No data in selected date range. Try expanding your date range.")
        else:
            # --- VISUAL ANALYTICS DASHBOARD ---
            st.markdown("### 📈 Performance Overview")
            
            # Prepare metrics
            dates_formatted = []
            completion_rates = []
            completed_counts = []
            total_counts = []
            energies = []
            
            for date, data in filtered_data.items():
                dt = datetime.strptime(date, "%Y-%m-%d")
                dates_formatted.append(dt.strftime("%b %d"))
                completed = len(data["completed"])
                total = len(data["all"])
                completed_counts.append(completed)
                total_counts.append(total)
                completion_rates.append(round(completed/total*100) if total > 0 else 0)
                
                # Get energy level from history if available
                history = load_json(HISTORY_FILE)
                energies.append(history.get(date, {}).get("energy", "Unknown"))
            
            # Metrics row
            avg_completion = round(sum(completion_rates)/len(completion_rates))
            best_day = max(filtered_data.items(), 
                         key=lambda x: len(x[1]["completed"])/len(x[1]["all"]))
            worst_day = min(filtered_data.items(), 
                          key=lambda x: len(x[1]["completed"])/len(x[1]["all"]))
     
            # Completion trend chart
            st.markdown("#### 📅 Daily Completion Rate")
            fig = px.line(
                x=dates_formatted,
                y=completion_rates,
                labels={"x": "Date", "y": "Completion %"},
                markers=True,
                height=350
            )
            fig.update_traces(line_color="#6e48aa", marker_color="#9d50bb")
            fig.update_layout(
                hovermode="x unified",
                xaxis_title="",
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- ENHANCED PREDICTIVE ANALYTICS ---
            st.markdown("## 🔮 Smart Predictions")
            with st.expander("🚀 Predict Future Productivity", expanded=True):
                
                st.info("""
                **How this works:**  
                Our AI analyzes your historical patterns to forecast how likely you are 
                to complete tasks based on day, energy level, and workload.
                """, icon="ℹ️")
                
                pred_col1, pred_col2 = st.columns([1, 1], gap="large")
                
                with pred_col1:
                    st.markdown("#### 🧠 Input Parameters")
                    day = st.selectbox(
                        "**Day of week**",
                        ["Monday", "Tuesday", "Wednesday", "Thursday", 
                         "Friday", "Saturday", "Sunday"],
                        index=2,  # Default to Wednesday
                        help="Productivity often varies by weekday"
                    )
                    
                    energy = st.radio(
                        "**Expected energy level**",
                        ["Low", "Medium", "High"],
                        index=1,
                        horizontal=True,
                        help="Your anticipated energy that day"
                    )
                
                with pred_col2:
                    st.markdown("#### 📝 Task Planning")
                    task_count = st.slider(
                        "**Number of tasks planned**",
                        min_value=1,
                        max_value=15,
                        value=6,
                        help="More tasks = lower completion probability"
                    )
                    
                    # Visual task load indicator
                    load_col1, load_col2, load_col3 = st.columns([1,3,1])
                    with load_col2:
                        if task_count <= 4:
                            st.success("😊 Light load")
                        elif task_count <= 7:
                            st.warning("😐 Moderate load")
                        else:
                            st.error("😟 Heavy load")
                
                if st.button("**Calculate My Success Probability**", 
                           type="primary", 
                           use_container_width=True):
                    
                    with st.spinner("🧠 Analyzing your productivity patterns..."):
                        try:
                            df = prepare_analytics_data()
                            
                            if df is None or len(df) < 9:
                                st.warning("""
                                ### 📊 More Data Needed
                                We need **10+ days** of tracked tasks for accurate predictions.
                                
                                **Tip:** Use the app daily for 2 weeks to unlock predictions.
                                """)
                                
                                # Show sample prediction
                                st.markdown("""
                                #### 🧐 Example Prediction (with sufficient data)
                                
                                | Factor | Value | Impact |
                                |--------|-------|--------|
                                | **Day** | Wednesday | +15% |
                                | **Energy** | Medium | +20% |
                                | **Tasks** | 6 | -10% |
                                | **Prediction** | 72% likely to complete most tasks |
                                """)
                                
                            else:
                                # Enhanced feature engineering
                                df["is_weekend"] = df["weekday"].apply(lambda x: x >= 5)
                                energy_map = {"Low": 0, "Medium": 1, "High": 2}
                                df["energy_encoded"] = df["energy"].map(energy_map)
                                
                                # Dynamic target based on user's median performance
                                median_rate = df["completion_rate"].median()
                                df["target"] = (df["completion_rate"] > median_rate).astype(int)
                                
                                X = df[["weekday", "is_weekend", "energy_encoded", "total_tasks"]]
                                y = df["target"]
                                
                                # Train optimized model
                                model = RandomForestClassifier(
                                    n_estimators=200,
                                    max_depth=5,
                                    class_weight='balanced',
                                    random_state=42
                                )
                                model.fit(X, y)
                                
                                # Prepare prediction input
                                weekday_map = {name: i for i, name in enumerate(
                                    ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                     "Friday", "Saturday", "Sunday"]
                                )}
                                
                                input_data = [[
                                    weekday_map[day],
                                    weekday_map[day] >= 5,
                                    energy_map[energy],
                                    task_count
                                ]]
                                
                                # Get probability prediction
                                proba = model.predict_proba(input_data)[0][1]
                                
                                # Display results with visual impact
                                st.markdown(f"""
                                ### 📊 Prediction Results
                                
                                <div style="background:#f8f9fa;padding:20px;border-radius:10px;text-align:center">
                                    <h2 style="color:#6e48aa;margin-bottom:5px">
                                        {proba*100:.0f}% Completion Probability
                                    </h2>
                                    <p style="font-size:16px">
                                        {"> 75% = 😊 Excellent day planned" if proba > 0.75 else 
                                         "60-75% = 🙂 Good potential" if proba > 0.6 else 
                                         "45-60% = 😐 Could be better" if proba > 0.45 else 
                                         "< 45% = 😟 High risk of incomplete tasks"}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Actionable recommendations
                                st.markdown("#### 💡 Optimization Tips")
                                rec_col1, rec_col2 = st.columns(2)
                                
                                with rec_col1:
                                    if proba < 0.5:
                                        st.error("""
                                        **🚨 High Risk Plan**  
                                        Consider:
                                        - Reducing to {0} tasks
                                        - Scheduling for {1}
                                        - Breaking large tasks down
                                        """.format(
                                            max(1, task_count-2),
                                            "Friday" if energy == "High" else "a higher-energy day"
                                        ))
                                    else:
                                        st.success("""
                                        **✅ Solid Plan**  
                                        Suggestions:
                                        - Add 15-minute buffers
                                        - Tackle hardest task first
                                        - Prep materials in advance
                                        """)
                                
                                with rec_col2:
                                    st.info("""
                                    **🔎 Key Insights**  
                                    Your productivity tends to be:
                                    - {0} on {1}s
                                    - {2} with {3} energy
                                    - {4} with {5} tasks
                                    """.format(
                                        "higher" if weekday_map[day] < 5 else "lower",
                                        day,
                                        "better" if energy == "High" else "worse",
                                        energy.lower(),
                                        "better" if task_count < 6 else "worse",
                                        task_count
                                    ))
                                
                                # Feature importance visualization
                                st.markdown("#### 🔍 What Impacts Your Productivity?")
                                importances = pd.DataFrame({
                                    'Factor': ["Day of Week", "Weekend", "Energy Level", "Task Volume"],
                                    'Impact': model.feature_importances_
                                }).sort_values('Impact', ascending=True)
                                
                                fig = px.bar(
                                    importances,
                                    x='Impact',
                                    y='Factor',
                                    orientation='h',
                                    color='Factor',
                                    color_discrete_sequence=px.colors.qualitative.Pastel,
                                    height=300
                                )
                                fig.update_layout(
                                    showlegend=False,
                                    xaxis_title="Relative Importance",
                                    yaxis_title="",
                                    margin=dict(l=20, r=20, t=30, b=20)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"""
                            ### ❌ Prediction Error
                            We couldn't generate a prediction because:
                            `{str(e)}`
                            
                            Please try again with different parameters.
                            """)
                            
                            # Debug view (hidden by default)
                            with st.expander("Technical Details (for support)"):
                                st.write("Error type:", type(e).__name__)
                                if 'df' in locals():
                                    st.write("Data shape:", df.shape if df is not None else "No data")
            
            # --- RAW DATA VIEW ---
            with st.expander("📁 View Raw Data"):
                st.dataframe(
                    pd.DataFrame({
                        'Date': dates_formatted,
                        'Completed': completed_counts,
                        'Total': total_counts,
                        'Rate (%)': completion_rates,
                        'Energy': energies
                    }),
                    height=300,
                    use_container_width=True
                )
            
# --- Task Recovery ---
elif section == "🔁 Task Recovery":
    st.markdown("### 🔄 Manage Missed Tasks")
    st.info("Review incomplete tasks and decide how to handle them", icon="ℹ️")
    
    completion = load_json(COMPLETION_FILE)
    history = load_json(HISTORY_FILE)
    
    if not completion:
        st.info("No completion data available yet.")
    else:
        tab1, tab2 = st.tabs(["📅 Reschedule Tasks", "🧠 Analyze Patterns"])
        
        with tab1:
            st.markdown("#### Reschedule Incomplete Tasks")
            
            # Get dates with incomplete tasks safely
            dates_with_incomplete = []
            for date in completion.keys():
                try:
                    if date in history and "input" in history[date]:
                        completed_tasks = set(completion[date].get("completed", []))
                        all_tasks = set(history[date]["input"])
                        if len(all_tasks - completed_tasks) > 0:
                            dates_with_incomplete.append(date)
                except KeyError:
                    continue
            
            if not dates_with_incomplete:
                st.success("No dates with incomplete tasks found!")
            else:
                selected = st.selectbox(
                    "Select date with incomplete tasks",
                    sorted(dates_with_incomplete, reverse=True),
                    format_func=lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%A, %B %d, %Y")
                )
                
                try:
                    all_tasks = history[selected]["input"]
                    completed = completion[selected]["completed"]
                    missed = list(set(all_tasks) - set(completed))
                    
                    if not missed:
                        st.success("No incomplete tasks for this date!")
                    else:
                        st.markdown(f"**Incomplete tasks from {selected}**")
                        tasks_to_reschedule = st.multiselect(
                            "Select tasks to reschedule",
                            missed,
                            default=missed
                        )
                        
                        if tasks_to_reschedule:
                            st.markdown("#### Reschedule Options")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                new_date = st.date_input("New target date", datetime.now().date() + timedelta(days=1))
                                new_energy = st.select_slider(
                                    "Expected energy level",
                                    ["Low", "Medium", "High"],
                                    value="Medium"
                                )
                            
                            with col2:
                                st.markdown("**Rescheduling Advice**")
                                st.info("""
                                - Morning: Best for high-focus tasks
                                - Afternoon: Good for medium-effort work
                                - Evening: Best for low-energy tasks
                                """)
                            
                            if st.button("🔄 Reschedule Tasks", type="primary"):
                                # Add to new date's plan
                                new_date_str = str(new_date)
                                if new_date_str not in history:
                                    history[new_date_str] = {
                                        "date": new_date_str,
                                        "energy": new_energy,
                                        "input": [],
                                        "manual_priorities": {"high": [], "medium": [], "low": []},
                                        "output": ""
                                    }
                                
                                # Add tasks avoiding duplicates
                                existing_tasks = set(history[new_date_str]["input"])
                                added_tasks = [t for t in tasks_to_reschedule if t not in existing_tasks]
                                history[new_date_str]["input"].extend(added_tasks)
                                
                                # Save changes
                                save_json(history, HISTORY_FILE)
                                st.success(f"Added {len(added_tasks)} tasks to {new_date_str}")
                                
                                # Remove from original if all were rescheduled
                                if set(missed) == set(tasks_to_reschedule):
                                    del completion[selected]
                                    save_json(completion, COMPLETION_FILE)
                
                except KeyError as e:
                    st.error(f"Error accessing data for selected date: {str(e)}")
        
        with tab2:
            st.markdown("#### Analyze Completion Patterns")
            if st.button("🧠 Generate Insights", type="primary"):
                with st.spinner("Analyzing your productivity patterns..."):
                    # Prepare data for analysis safely
                    analysis_data = []
                    for date in completion.keys():
                        try:
                            if date in history and "input" in history[date]:
                                total = len(history[date]["input"])
                                completed = len(completion[date].get("completed", []))
                                missed = total - completed
                                energy = history[date].get("energy", "Unknown")
                                analysis_data.append({
                                    "date": date,
                                    "energy": energy,
                                    "total": total,
                                    "completed": completed,
                                    "missed": missed,
                                    "rate": round(completed/total*100) if total > 0 else 0
                                })
                        except KeyError:
                            continue
                    
                    if not analysis_data:
                        st.warning("No valid data available for analysis")
                    else:
                        prompt = f"""
Analyze this productivity data and provide insights:

{json.dumps(analysis_data, indent=2)}

Look for:
1. Patterns between energy levels and completion rates
2. Days of week with best/worst performance
3. Task load vs completion correlation
4. Personalized recommendations

Provide the analysis in markdown format with:
- Key findings
- Visual patterns
- Actionable advice
"""
                        try:
                            response = model.generate_content(prompt)
                            st.markdown(response.text, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")

# --- Share Plan ---
elif section == "📤 Share Plan":
    st.markdown("### 📤 Export & Share Plans")
    history = load_json(HISTORY_FILE)
    
    if not history:
        st.info("No plans available to share")
    else:
        selected = st.selectbox(
            "Select plan to share",
            sorted(history.keys(), reverse=True),
            format_func=lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%A, %B %d, %Y")
        )
        
        plan = history[selected]
        
        st.markdown("#### Share Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📄 Text Format**")
            st.download_button(
                "Download TXT",
                plan['output'],
                file_name=f"productivity_plan_{selected}.txt",
                mime="text/plain"
            )
        
        with col2:
            st.markdown("**📊 PDF Report**")
            pdf_report = create_pdf_report(
                content=plan['output'],
                title=f"Productivity Report - {selected}"
            )
            st.download_button(
                "Download PDF",
                data=pdf_report,
                file_name=f"productivity_report_{selected}.pdf",
                mime="application/pdf"
            )
        
        with col3:
            st.markdown("**📅 Calendar**")
            st.button("Export to Calendar (Coming Soon)", disabled=True)
        
        st.markdown("#### Preview")
        with st.expander("View Plan Content"):
            st.markdown(plan['output'], unsafe_allow_html=True)

# --- Daily Reflection ---
elif section == "📄 Daily Reflection":
    st.markdown("### 📝 Daily Reflection")
    st.info("Review your day and gain insights for improvement", icon="🧠")
    
    completion = load_json(COMPLETION_FILE)
    history = load_json(HISTORY_FILE)
    
    if not completion:
        st.info("No completion data available yet")
    else:
        selected = st.selectbox(
            "Select date to reflect on",
            sorted(completion.keys(), reverse=True),
            format_func=lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%A, %B %d, %Y")
        )
        
        # Get completion data
        comp_data = completion[selected]
        completed = comp_data["completed"]
        total = len(comp_data["all"])
        missed = list(set(comp_data["all"]) - set(completed))
        
        # Get original plan data
        plan_data = history.get(selected, {})
        energy = plan_data.get("energy", "Unknown")
        
        # Reflection form
        with st.form("reflection_form"):
            st.markdown("#### How was your day?")
            mood = st.radio(
                "Overall mood",
                ["😊 Great", "🙂 Good", "😐 Okay", "😕 Tough", "😞 Difficult"],
                horizontal=True
            )
            
            productivity = st.slider(
                "Productivity level", 
                1, 10, 7,
                help="How productive were you overall?"
            )
            
            distractions = st.text_area(
                "What distracted you today?",
                placeholder="Social media, unexpected events, fatigue...",
                height=100
            )
            
            lessons = st.text_area(
                "What did you learn today?",
                placeholder="About your productivity, energy patterns, etc.",
                height=100
            )
            
            submitted = st.form_submit_button("✨ Generate Reflection", type="primary")
            
            if submitted:
                with st.spinner("Generating personalized reflection..."):
                    prompt = f"""
Create a daily reflection based on this data:

**Date:** {selected}
**Energy Level:** {energy}
**Mood:** {mood}
**Productivity:** {productivity}/10
**Completion Rate:** {len(completed)}/{total} ({round(len(completed)/total*100)}%)
**Missed Tasks:** {missed if missed else "None"}
**Distractions:** {distractions if distractions else "None reported"}
**Lessons:** {lessons if lessons else "None reported"}

Generate a thoughtful reflection with:
1. **Achievements**: Highlight successes
2. **Challenges**: What went wrong
3. **Insights**: Patterns noticed
4. **Recommendations**: For tomorrow
5. **Encouragement**: Positive reinforcement

Use a warm, coaching tone. Output in markdown.
"""
                    try:
                        response = model.generate_content(prompt)
                        st.markdown("### 📝 Your Daily Reflection")
                        st.markdown(response.text, unsafe_allow_html=True)
                        # ===== NEW HABIT FORMATION CODE =====
                        st.markdown("---")
                        st.markdown("### 🔥 Personalized Habit Advice")
                        
                        habit_prompt = f"""
                        Based on this daily reflection:
                        Mood: {mood}
                        Productivity: {productivity}/10
                        Distractions: {distractions}
                        Lessons: {lessons}
                        Completion Rate: {len(completed)}/{total} tasks

                        Suggest 1-2 keystone habits using these rules:
                        1. Focus on small, actionable habits (max 20 words each)
                        2. Format as: "✨ [Habit] (Science: [psychology principle])"
                        3. Reference: Atomic Habits, Tiny Habits, or Stanford Persuasive Tech Lab
                        
                        Example: 
                        ✨ Drink water before checking phone (Science: Implementation intention)
                        """
                        
                        habit_response = model.generate_content(habit_prompt)
                        st.markdown(habit_response.text)
                        
                        # Save to history
                        if "reflections" not in history[selected]:
                            history[selected]["reflections"] = []
                            
                        history[selected]["reflections"].append({
                            # ... [keep existing reflection save code] ...
                            "habit_advice": habit_response.text  # Add this new field
                        })
                           
                    except Exception as e:
                        st.error(f"Failed to generate reflection: {str(e)}")

# --- Footer ---
st.markdown("""
<hr style='margin-top:50px;'>
<div style='text-align:center; color:#666; font-size:0.9em;'>
    Krisha Goti
</div>
""", unsafe_allow_html=True)