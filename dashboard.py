import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Configuration ---
LOG_FILE = 'attendance/attendance_log.csv'

st.set_page_config(layout="wide", page_title="Attendance System Dashboard")

def load_data():
    """Load the attendance log file."""
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    return pd.DataFrame(columns=['Timestamp', 'Name', 'Status'])

# --- Dashboard Layout ---
st.title("Attendance System Report 📊")
st.markdown("Real-time snapshot of the biometric attendance log.")

df = load_data()

if df.empty:
    st.warning(f"No attendance data found in {LOG_FILE}. Please run the recognition script first.")
else:
    # --- 1. Key Metrics ---
    col1, col2, col3 = st.columns(3)
    
    unique_attendees = df['Name'].nunique()
    total_records = len(df)
    last_checkin_time = df['Timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
    col1.metric("Unique Attendees", unique_attendees)
    col2.metric("Total Check-Ins", total_records)
    col3.metric("Last Check-In", last_checkin_time)

    st.markdown("---")

    # --- 2. Attendance Table ---
    st.header("Attendance Log (Last 100 Entries)")
    st.dataframe(df.tail(100).sort_values(by='Timestamp', ascending=False), width='stretch')

    st.markdown("---")

    # --- 3. Visualizations ---
    st.header("Attendance Statistics")
    
    # Chart: Check-in Counts by Person
    st.subheader("Check-Ins per Person")
    person_counts = df['Name'].value_counts().sort_index()
    st.bar_chart(person_counts)

    # Chart: Check-ins Over Time (Hourly)
    df['Hour'] = df['Timestamp'].dt.hour
    hourly_counts = df.groupby('Hour')['Name'].count()
    st.subheader("Hourly Check-In Frequency")
    st.line_chart(hourly_counts)