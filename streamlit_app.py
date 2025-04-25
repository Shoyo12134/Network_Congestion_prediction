import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from PIL import Image  # Added for simulation images

# -----------------------------------------------------------------------------
# Dashboard Title and Overview
# -----------------------------------------------------------------------------
st.title("Network Congestion Dashboard")
st.write("""
This dashboard displays network congestion predictions and includes two main sections:
1. **Static Predictions Analysis**: Upload your prediction file (CSV) to review and analyze the predictions.
2. **Real Time Simulation**: See a live simulation of network congestion predictions with routing decisions.
""")

# -----------------------------------------------------------------------------
# Section 1: Static Predictions Analysis
# -----------------------------------------------------------------------------
st.header("Static Predictions Analysis")

# Allow the user to upload the predictions CSV file generated from your prediction script.
uploaded_file = st.file_uploader("Upload the balanced predictions CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(uploaded_file)
    
    # Display the first few rows so the user can verify the data.
    st.write("### Predictions Data (Sample):")
    st.write(df.head())
    
    # Display summary statistics such as count, mean, min, and max for the numerical data.
    st.write("### Summary Statistics:")
    st.write(df.describe())
    
    # Plot the distribution of the continuous congestion level.
    st.write("### Distribution of Continuous Congestion Level:")
    fig, ax = plt.subplots()
    sns.histplot(df['Continuous Congestion Level'], bins=20, kde=True, ax=ax)
    ax.set_title("Distribution of Continuous Congestion Level")
    ax.set_xlabel("Congestion Level")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    
    # Countplot for binary predictions (0 = no congestion, 1 = congestion)
    st.write("### Binary Congestion Prediction Counts:")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="Binary Congestion", data=df, ax=ax2)
    ax2.set_title("Binary Congestion Predictions (0 = No, 1 = Yes)")
    ax2.set_xlabel("Binary Congestion")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)
    
    # Countplot for load balancing decisions.
    st.write("### Load Balancer Routing Decisions:")
    fig3, ax3 = plt.subplots()
    sns.countplot(x="Routing Decision", data=df, ax=ax3)
    ax3.set_title("Load Balancer Routing Decisions")
    ax3.set_xlabel("Decision")
    ax3.set_ylabel("Count")
    st.pyplot(fig3)
    
    # If there is an 'Anomaly' column, display its counts.
    if 'Anomaly' in df.columns:
        st.write("### Anomaly Detection:")
        anomaly_counts = df['Anomaly'].value_counts()
        st.write(anomaly_counts)
else:
    st.info("Upload a balanced predictions CSV file to view static analysis.")

# -----------------------------------------------------------------------------
# Section 2: Real Time Simulation
# -----------------------------------------------------------------------------
st.header("Real Time Simulation")
st.write("""
This section simulates real-time network congestion predictions.
- **Binary Congestion**: 0 means no congestion, 1 means congestion.
- **Continuous Congestion Level**: A numeric value (0–100) indicating how severe the congestion is.
- **Routing Decision**: If there is congestion or if the congestion level is high (≥ 70), the traffic is routed to a backup server; otherwise, it goes via the primary server.
""")

# Toggle for enabling real-time simulation.
simulate_real_time = st.checkbox("Enable Real Time Simulation", value=False)

if simulate_real_time:
    # User inputs to determine how long the simulation will run and how often it updates.
    simulation_duration = st.number_input("Simulation Duration (seconds):", min_value=10, max_value=300, value=30, step=10)
    simulation_interval = st.number_input("Update Interval (seconds):", min_value=1, max_value=10, value=3, step=1)
    
    st.write("Click the button below to start the real time simulation.")
    start_simulation = st.button("Start Simulation")
    
    if start_simulation:
        # Create placeholders in the dashboard for updating simulation results, plots, and images.
        simulation_placeholder = st.empty()  # For displaying simulation data table.
        plot_placeholder = st.empty()          # For displaying the real-time congestion plot.
        simulation_img_placeholder = st.empty() # For displaying simulation images based on routing decisions.
        
        # Create an empty DataFrame to store simulation records.
        simulation_data = pd.DataFrame(columns=[
            'Timestamp', 'Binary Congestion', 'Continuous Congestion Level', 'Routing Decision'
        ])
        
        start_time = time.time()
        # Run the simulation loop until the specified simulation duration is reached.
        while (time.time() - start_time) < simulation_duration:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Simulate binary congestion (0: no congestion, 1: congestion). Here a 30% chance for congestion.
            binary_congestion = np.random.choice([0, 1], p=[0.7, 0.3])
            # Simulate a continuous congestion level between 0 and 100.
            continuous_congestion = np.random.uniform(0, 100)
            
            # Decision logic: if congestion exists or the level is high (>=70), use a backup server.
            threshold = 70
            if binary_congestion == 1 or continuous_congestion >= threshold:
                decision = "Route to Backup Server"
            else:
                decision = "Route via Primary Server"
            
            # Create a record for the current simulation step.
            new_record = {
                'Timestamp': current_time,
                'Binary Congestion': binary_congestion,
                'Continuous Congestion Level': continuous_congestion,
                'Routing Decision': decision
            }
            
            # Append the new record to the simulation data using pd.concat
            simulation_data = pd.concat(
                [simulation_data, pd.DataFrame([new_record])],
                ignore_index=True
            )
            
            # Update the simulation table to display the last 10 records.
            simulation_placeholder.write(simulation_data.tail(10))
            
            # Update the real-time line plot for continuous congestion levels.
            fig_rt, ax_rt = plt.subplots()
            sns.lineplot(x=simulation_data.index, y='Continuous Congestion Level', data=simulation_data, ax=ax_rt)
            ax_rt.set_title("Real Time Continuous Congestion Level")
            ax_rt.set_xlabel("Record Number")
            ax_rt.set_ylabel("Congestion Level")
            plot_placeholder.pyplot(fig_rt)
            
            # -----------------------------------------------------------------
            # Generate and display an image based on the routing decision.
            # -----------------------------------------------------------------
            # Use a red image to represent routing to the backup server (high congestion)
            # and a green image for routing via the primary server (low congestion).
            if decision == "Route to Backup Server":
                color = (255, 0, 0)  # Red
            else:
                color = (0, 255, 0)  # Green
            img = Image.new("RGB", (300, 200), color)
            simulation_img_placeholder.image(img, caption=f"Current Decision: {decision}", use_column_width=True)
            
            # Pause for the specified interval before the next simulation update.
            time.sleep(simulation_interval)
        
        st.success("Real Time Simulation Completed!")