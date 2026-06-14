import streamlit as st
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import textwrap
import tempfile

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="KI Reithaltungs-Analyse", layout="wide")

st.title("🏇 KI Reithaltungs-Analyse")
st.markdown("Lade dein Video hoch, um die vollständige Analyse inklusive Bewertungsbögen zu erhalten.")

# --- DATEI-UPLOAD ---
email = st.text_input("Deine E-Mail-Adresse:", placeholder="reiter@beispiel.de")
uploaded_file = st.file_uploader("Wähle ein Video aus", type=['mp4', 'mov', 'avi'])

if uploaded_file and email:
    if st.button("Komplette Analyse starten"):
        # Temporäre Dateien für Ein- und Ausgabe
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_eingabe = tfile.name
        
        video_ausgabe = video_eingabe.replace('.mp4', '_analyse_ergebnis.mp4')
        hintergrund_bild = "reining_hintergrund.jpg" # Optional, falls vorhanden

        # ==============================================================================
        # DEIN ORIGINAL-CODE STARTET HIER (1:1 ÜBERNOMMEN)
        # ==============================================================================
        
        print("Lade YOLO-Pose Modell...")
        model = YOLO("yolov8n-pose.pt")

        if not os.path.exists(video_eingabe):
            st.error(f"Datei '{video_eingabe}' nicht gefunden!")
        else:
            cap = cv2.VideoCapture(video_eingabe)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = cap.get(cv2.CAP_PROP_FPS)
            is_portrait = height > width

            if fps > 0:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                total_frames = 1

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_ausgabe, fourcc, fps, (width, height))

            # Datenspeicher
            sitz_daten_gesamt, blick_daten_gesamt, hand_daten_gesamt, bein_daten_gesamt = [], [], [], []
            sitz_historie, blick_historie, hand_historie, bein_historie = [], [], [], []
            hip_y_buffer, trab_historie = [], []
            
            # EMA Glättung
            sm_sx, sm_sy, sm_hx, sm_hy, sm_nx, sm_ny, sm_ex, sm_ey, sm_hax, sm_hay, sm_kx, sm_ky, sm_fx, sm_fy = [None]*14
            alpha = 0.3

            def get_color(score):
                if score > 5: return (0, 255, 0)
                elif score == 5: return (0, 255, 255)
                else: return (0, 0, 255)

            progress_bar = st.progress(0)
            status_text = st.empty()

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame_count += 1
                
                if frame_count % 10 == 0:
                    progress_bar.progress(frame_count / total_frames)
                    status_text.text(f"Analysiere Frame {frame_count} von {total_frames}...")

                results = model(frame, verbose=False)[0]
                
                if results.keypoints is not None and len(results.keypoints.data) > 0:
                    for i, kpts_raw in enumerate(results.keypoints.data):
                        kpts = kpts_raw.cpu().numpy()
                        box_xyxy = results.boxes.xyxy[i].cpu().numpy()
                        bx1, by1, bx2, by2 = int(box_xyxy[0]), int(box_xyxy[1]), int(box_xyxy[2]), int(box_xyxy[3])

                        if by1 < (height * 0.6) and (by2 - by1) > 40:
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

                            if kpts[5][2] > 0.3 and kpts[11][2] > 0.3:
                                raw_sx, raw_sy = (kpts[5][0] + kpts[6][0]) / 2.0, (kpts[5][1] + kpts[6][1]) / 2.0
                                raw_hx, raw_hy = (kpts[11][0] + kpts[12][0]) / 2.0, (kpts[11][1] + kpts[12][1]) / 2.0
                                
                                if sm_sx is None:
                                    sm_sx, sm_sy, sm_hx, sm_hy = raw_sx, raw_sy, raw_hx, raw_hy
                                else:
                                    sm_sx, sm_sy = raw_sx * alpha + sm_sx * (1 - alpha), raw_sy * alpha + sm_sy * (1 - alpha)
                                    sm_hx, sm_hy = raw_hx * alpha + sm_hx * (1 - alpha), raw_hy * alpha + sm_hy * (1 - alpha)

                                body_scale = max(1.0, math.sqrt((sm_sx - sm_hx)**2 + (sm_sy - sm_hy)**2))
                                
                                hip_y_buffer.append(sm_hy / body_scale)
                                if len(hip_y_buffer) > 30: hip_y_buffer.pop(0)
                                ist_trab = False
                                if len(hip_y_buffer) == 30:
                                    local_mean = sum(hip_y_buffer) / 30.0
                                    variance = sum((y - local_mean)**2 for y in hip_y_buffer) / 30.0
                                    if variance > 0.0015: ist_trab = True

                                # SCORES
                                s_score = max(1, min(10, int(10 - (abs(sm_sx - sm_hx) / body_scale * 15))))
                                
                                if kpts[0][2] > 0.3:
                                    raw_nx, raw_ny = kpts[0][0], kpts[0][1]
                                    sm_nx = raw_nx if sm_nx is None else raw_nx * alpha + sm_nx * (1 - alpha)
                                    b_score = max(1, min(10, int(10 - (abs(sm_nx - sm_sx) / body_scale * 10))))
                                else: b_score = 5

                                if kpts[7][2] > 0.3 and kpts[9][2] > 0.3:
                                    raw_ey, raw_hay = kpts[7][1], kpts[9][1]
                                    sm_ey = raw_ey if sm_ey is None else raw_ey * alpha + sm_ey * (1 - alpha)
                                    sm_hay = raw_hay if sm_hay is None else raw_hay * alpha + sm_hay * (1 - alpha)
                                    h_score = max(1, min(10, int(10 - (abs(sm_ey - sm_hay) / body_scale * 12))))
                                else: h_score = 5

                                if kpts[13][2] > 0.3 and kpts[15][2] > 0.3:
                                    raw_kx, raw_fx = kpts[13][0], kpts[15][0]
                                    sm_kx = raw_kx if sm_kx is None else raw_kx * alpha + sm_kx * (1 - alpha)
                                    sm_fx = raw_fx if sm_fx is None else raw_fx * alpha + sm_fx * (1 - alpha)
                                    be_score = max(1, min(10, int(10 - (abs(sm_kx - sm_fx) / body_scale * 10))))
                                else: be_score = 5

                                sitz_daten_gesamt.append(s_score)
                                blick_daten_gesamt.append(b_score)
                                hand_daten_gesamt.append(h_score)
                                bein_daten_gesamt.append(be_score)

                                cv2.putText(frame, f"SITZ: {s_score}", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, get_color(s_score), 2)
                                if ist_trab: cv2.putText(frame, "MODUS: TRAB", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

                out.write(frame)

            # --- ABSCHLUSS-GRAPHEN (DEIN ORIGINAL-CODE) ---
            if len(sitz_daten_gesamt) > 0:
                fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
                fig.patch.set_facecolor('#0F0F0F')
                ax.set_facecolor('#0F0F0F')
                x_achse = [i/fps for i in range(len(sitz_daten_gesamt))]
                ax.plot(x_achse, pd.Series(sitz_daten_gesamt).ewm(alpha=0.4).mean(), label="Sitz", color="#00FF66", lw=2.5)
                ax.plot(x_achse, pd.Series(blick_daten_gesamt).ewm(alpha=0.4).mean(), label="Blick", color="#00E5FF", lw=2.5)
                ax.plot(x_achse, pd.Series(hand_daten_gesamt).ewm(alpha=0.4).mean(), label="Hand", color="#FF9100", lw=2.5)
                ax.plot(x_achse, pd.Series(bein_daten_gesamt).ewm(alpha=0.4).mean(), label="Bein", color="#D500F9", lw=2.5)
                ax.set_title("DEINE LEISTUNGS-ANALYSE", color='white', fontsize=20, fontweight='bold')
                ax.set_ylim(0.5, 10.5)
                ax.legend(loc="upper right", framealpha=0.2, facecolor='black', labelcolor='white')
                
                temp_graph = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
                plt.savefig(temp_graph)
                plt.close()
                
                graph_img = cv2.imread(temp_graph)
                graph_img = cv2.resize(graph_img, (width, height))
                for _ in range(int(fps * 5)):
                    out.write(graph_img)

            cap.release()
            out.release()
            
            st.success("Analyse abgeschlossen!")
            st.video(video_ausgabe)
            
            # Cleanup
            os.remove(video_eingabe)
else:
    st.info("Bitte gib deine E-Mail an und lade ein Video hoch.")