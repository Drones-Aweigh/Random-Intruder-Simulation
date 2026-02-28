import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Random Intruder Simulation", layout="wide")
st.title("Random Intruder Collision Risk Simulation")
st.markdown("""
**Computational approach to calculating the risk of two objects colliding in 3D space**  
*Developed by Dr. David Lemon (Center for Naval Analyses) and Dr. Christopher P. Heagney (NAVAIR)*  
Revision: 18 August 2025 – Fully interactive Streamlit implementation
""")

# ====================== SIDEBAR PARAMETERS ======================
with st.sidebar:
    st.header("Simulation Controls")
    
    run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    st.subheader("Core Parameters")
    max_rays = st.number_input("Maximum Rays", min_value=1_000, max_value=500_000, value=50_000, step=5_000,
                               help="Higher values increase statistical confidence but take longer")
    max_collisions = st.number_input("Stop after N Collisions (0 = run all rays)", min_value=0, max_value=500, value=50, step=1)
    
    st.subheader("Airspace Volume")
    xy_min = st.number_input("xy_min", min_value=50, max_value=2_000, value=250, step=10)
    xy_max = st.number_input("xy_max", min_value=100, max_value=10_000, value=750, step=10)
    z_min = st.number_input("z_min", min_value=50, max_value=2_000, value=100, step=10)
    z_max = st.number_input("z_max", min_value=100, max_value=10_000, value=600, step=10)
    
    st.subheader("Ray Properties")
    max_ray_length = st.number_input("Max Ray Length (units)", min_value=100, max_value=10_000, value=1_580, step=10)
    max_ray_hours = st.number_input("Max Ray Duration (hours)", min_value=1, max_value=100, value=10, step=1)
    ray_radius = st.number_input("Ray Radius (half wingspan)", min_value=5, max_value=200, value=12, step=1)
    
    batch_samples = st.slider("Number of Volume Batches", 1, 20, 1)
    
    st.subheader("Visualization")
    show_3d = st.checkbox("Show interactive 3D view of last collision", value=True)
    st.caption("Only the final collision is visualized to keep performance high.")

# ====================== SIMULATION LOGIC ======================
if run_button:
    if max_collisions == 0:
        max_collisions = -1
    
    simulation_start = datetime.now()
    intersections_per_batch = []
    colliding_rays_info = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for batch in range(batch_samples):
        if batch_samples > 1:
            xy_limit = xy_min + (xy_max - xy_min) * batch / (batch_samples - 1)
            z_limit = z_min + (z_max - z_min) * batch / (batch_samples - 1)
        else:
            xy_limit = xy_min
            z_limit = z_min
        
        ray_counter = 0
        intersect_counter = 0
        collision_counter = 0
        
        while (max_collisions == -1 or collision_counter < max_collisions) and (ray_counter < max_rays):
            ray_counter += 1
            
            # Static ray (ownship) – always along Z for simplicity as in original
            ray_length = np.random.uniform(ray_radius * 2, max_ray_length)
            static_end = np.array([0, 0, ray_length])
            
            # Random intruder ray (6 face entry as in original)
            case = np.random.randint(0, 6)
            if case == 0:   # left face
                random_origin = np.array([-xy_limit, np.random.uniform(-xy_limit, xy_limit), np.random.uniform(-z_limit, z_limit)])
                theta = np.random.uniform(-np.pi/2, np.pi/2)
                phi = np.random.uniform(0, np.pi)
            elif case == 1: # right face
                random_origin = np.array([xy_limit, np.random.uniform(-xy_limit, xy_limit), np.random.uniform(-z_limit, z_limit)])
                theta = np.random.uniform(np.pi/2, 3*np.pi/2)
                phi = np.random.uniform(0, np.pi)
            elif case == 2: # front face
                random_origin = np.array([np.random.uniform(-xy_limit, xy_limit), -xy_limit, np.random.uniform(-z_limit, z_limit)])
                theta = np.random.uniform(0, np.pi)
                phi = np.random.uniform(0, np.pi)
            elif case == 3: # back face
                random_origin = np.array([np.random.uniform(-xy_limit, xy_limit), xy_limit, np.random.uniform(-z_limit, z_limit)])
                theta = np.random.uniform(np.pi, 2*np.pi)
                phi = np.random.uniform(0, np.pi)
            elif case == 4: # bottom face
                random_origin = np.array([np.random.uniform(-xy_limit, xy_limit), np.random.uniform(-xy_limit, xy_limit), -z_limit])
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.random.uniform(0, np.pi/2)
            else:           # top face
                random_origin = np.array([np.random.uniform(-xy_limit, xy_limit), np.random.uniform(-xy_limit, xy_limit), z_limit])
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.random.uniform(np.pi/2, np.pi)
            
            random_end = random_origin + np.array([
                ray_length * np.sin(phi) * np.cos(theta),
                ray_length * np.sin(phi) * np.sin(theta),
                ray_length * np.cos(phi)
            ])
            
            # Reuse original distance function exactly
            def shortest_distance_between_lines(P1, D1, P2, D2):
                dir1 = D1 - P1
                dir2 = D2 - P2
                r = P1 - P2
                a = np.dot(dir1, dir1)
                b = np.dot(dir1, dir2)
                c = np.dot(dir1, r)
                e = np.dot(dir2, dir2)
                f = np.dot(dir2, r)
                denom = a * e - b * b
                s = 0.0
                t = 0.0
                if denom != 0:
                    s = np.clip((b * f - c * e) / denom, 0, 1)
                t = np.clip((b * s + f) / e, 0, 1)
                if denom != 0:
                    s = np.clip((b * t + c) / a, 0, 1)
                pt1 = P1 + dir1 * s
                pt2 = P2 + dir2 * t
                return pt1, pt2, np.linalg.norm(pt1 - pt2)
            
            pt1, pt2, distance = shortest_distance_between_lines(
                np.array([0,0,0]), static_end, random_origin, random_end)
            
            if distance < ray_radius * 2:
                intersect_counter += 1
                
                # Exact original collision-percentage logic
                static_len = np.linalg.norm(static_end)
                random_len = np.linalg.norm(random_end - random_origin)
                static_pct = np.linalg.norm(pt1) / static_len
                random_pct = np.linalg.norm(pt2 - random_origin) / random_len
                tol_static = 2 * ray_radius / static_len
                tol_random = 2 * ray_radius / random_len
                
                if not ((static_pct + tol_static) < (random_pct - tol_random) or
                        (random_pct + tol_random) < (static_pct - tol_static)):
                    collision_counter += 1
                    colliding_rays_info.append({
                        "Batch": batch + 1,
                        "Ray #": ray_counter,
                        "Distance": round(distance, 4),
                        "Static End": static_end.tolist(),
                        "Random Origin": random_origin.tolist(),
                        "Random End": random_end.tolist(),
                        "Closest Static": pt1.tolist(),
                        "Closest Random": pt2.tolist()
                    })
            
            # Progress update
            progress = min(1.0, ray_counter / max_rays)
            progress_bar.progress(progress)
            status_text.text(f"Batch {batch+1}/{batch_samples} • Rays: {ray_counter:,} • Intersects: {intersect_counter:,} • Collisions: {collision_counter}")
        
        # Store batch summary
        flight_hours = ray_counter * max_ray_hours
        collisions_per_hour = collision_counter / flight_hours if flight_hours > 0 else 0
        intersections_per_batch.append([
            round(xy_limit, 2), round(z_limit, 2), max_ray_length, ray_radius,
            ray_counter, intersect_counter, collision_counter,
            flight_hours, round(collisions_per_hour, 10)
        ])
    
    progress_bar.empty()
    status_text.empty()
    
    total_time = datetime.now() - simulation_start
    
    # ====================== RESULTS ======================
    st.success(f"Simulation completed in {total_time}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rays Simulated", f"{ray_counter:,}")
    with col2:
        st.metric("Total Intersections", f"{intersect_counter:,}")
    with col3:
        st.metric("Total Collisions", f"{collision_counter:,}")
    with col4:
        st.metric("Collision Rate", f"{collisions_per_hour:.2e} per flight hour")
    
    # Summary table
    df_summary = pd.DataFrame(intersections_per_batch, columns=[
        "xy_limit", "z_limit", "max_ray_length", "ray_radius",
        "Rays", "Intersects", "Collisions", "Flight Hours", "Collisions per Flight Hour"
    ])
    st.subheader("Batch Summary")
    st.dataframe(df_summary, use_container_width=True)
    
    # Detailed colliding rays
    if colliding_rays_info:
        df_collisions = pd.DataFrame(colliding_rays_info)
        st.subheader(f"Colliding Rays Details ({len(df_collisions)} events)")
        st.dataframe(df_collisions, use_container_width=True)
        
        # CSV downloads
        csv_summary = df_summary.to_csv(index=False).encode()
        csv_details = df_collisions.to_csv(index=False).encode()
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("📥 Download Summary CSV", csv_summary, f"intruder_simulation_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
        with col_dl2:
            st.download_button("📥 Download Collision Details CSV", csv_details, f"intruder_collisions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
    
    # Optional 3D Plot of LAST collision
    if show_3d and colliding_rays_info:
        last = colliding_rays_info[-1]
        fig = go.Figure()
        
        # Static ray (ownship)
        fig.add_trace(go.Scatter3d(x=[0, last["Static End"][0]],
                                   y=[0, last["Static End"][1]],
                                   z=[0, last["Static End"][2]],
                                   mode='lines+markers', line=dict(color='blue', width=6), name='Ownship (Static)'))
        
        # Random intruder ray
        ro = last["Random Origin"]
        re = last["Random End"]
        fig.add_trace(go.Scatter3d(x=[ro[0], re[0]], y=[ro[1], re[1]], z=[ro[2], re[2]],
                                   mode='lines+markers', line=dict(color='gray', width=6), name='Intruder (Random)'))
        
        # Closest points
        fig.add_trace(go.Scatter3d(x=[last["Closest Static"][0]], y=[last["Closest Static"][1]], z=[last["Closest Static"][2]],
                                   mode='markers', marker=dict(size=8, color='red'), name='Closest Point (Static)'))
        fig.add_trace(go.Scatter3d(x=[last["Closest Random"][0]], y=[last["Closest Random"][1]], z=[last["Closest Random"][2]],
                                   mode='markers', marker=dict(size=8, color='red'), name='Closest Point (Random)'))
        
        fig.update_layout(title="3D View of Last Detected Collision", scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), height=600)
        st.plotly_chart(fig, use_container_width=True)

st.caption("© 2026 – Streamlit adaptation of the original Random Intruder Simulation. All physics and collision logic are identical to the 18 August 2025 version.")
