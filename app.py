import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from scipy.interpolate import interp1d
from fpdf import FPDF
import tempfile
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
st.set_page_config(page_title="Análisis Granulométrico", layout="centered")
st.title("📷 Granulometría por Visión Artificial")

DENSIDAD_SOLIDO = 2.65
mallas = {
    '5"': 127.0, '4"': 101.6, '3"': 76.2, '2"': 50.8, '1.5"': 38.1, '1"': 25.4, '3/4"': 19.05,
    '1/2"': 12.7, '3/8"': 9.51, '1/4"': 6.35, '#4': 4.76, '#8': 2.38,
    '#14': 1.41, '#28': 0.595, '#48': 0.297, '#100': 0.149, '#200': 0.074
}

# ==========================================
# INTERFAZ
# ==========================================
uploaded_file = st.file_uploader("Toma una foto o sube una imagen de las rocas", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    ANCHO_PAPEL_CM = st.number_input("Ancho del papel ref. (cm)", min_value=0.1, value=14.8, step=0.1)
with col2:
    LARGO_PAPEL_CM = st.number_input("Largo del papel ref. (cm)", min_value=0.1, value=21.0, step=0.1)

if uploaded_file is not None:
    # 1. Leer imagen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Para mostrar en Streamlit
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    st.image(img_rgb, caption="Imagen original", use_column_width=True)
    
    if st.button("Procesar y Generar Reporte"):
        with st.spinner('Analizando imagen y calculando segmentación...'):
            
            # 2. MEJORA: Detectar Escala y Validación Visual
            _, thresh_papel = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
            contornos_papel, _ = cv2.findContours(thresh_papel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contornos_papel:
                st.error("No se detectó el papel. Asegúrate de que esté bien iluminado y sin demasiadas sombras.")
                st.stop()
                
            contorno_mayor = max(contornos_papel, key=cv2.contourArea)
            rect = cv2.minAreaRect(contorno_mayor)
            largo_px = max(rect[1])
            px_por_cm = largo_px / LARGO_PAPEL_CM
            
            # Dibujar el contorno detectado para que el usuario lo vea
            img_verificacion = img_rgb.copy()
            box = cv2.boxPoints(rect)
            box = np.int32(box) # Corregido np.int0 que está deprecado
            cv2.drawContours(img_verificacion, [box], 0, (0, 255, 0), 6)
            st.success("¡Referencia detectada!")
            st.image(img_verificacion, caption="Caja verde indica el papel detectado. Si no coincide, mejora la iluminación.", use_column_width=True)

            # 3. MEJORA: CLAHE para rocas oscuras + Watershed
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl_img = clahe.apply(img_gray)
            blur = cv2.GaussianBlur(cl_img, (7, 7), 0)
            
            _, thresh_rocas = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            dist_transform = ndimage.distance_transform_edt(thresh_rocas)
            local_max = peak_local_max(dist_transform, min_distance=20, labels=thresh_rocas)
            mask = np.zeros(dist_transform.shape, dtype=bool)
            mask[tuple(local_max.T)] = True
            marcadores, _ = ndimage.label(mask)
            labels = watershed(-dist_transform, marcadores, mask=thresh_rocas)

            # 4. Cálculo
            diametros_mm, masas_g = [], []
            for label in np.unique(labels):
                if label == 0: continue
                mask_roca = np.zeros(img_gray.shape, dtype="uint8")
                mask_roca[labels == label] = 255
                cnts, _ = cv2.findContours(mask_roca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(cnts) == 0: continue
                c = cnts[0]
                area_px = cv2.contourArea(c)
                if area_px < 50: continue
                
                area_cm2 = area_px / (px_por_cm ** 2)
                diametro_cm = 2 * np.sqrt(area_cm2 / np.pi)
                diametros_mm.append(diametro_cm * 10)
                volumen_cm3 = (4/3) * np.pi * ((diametro_cm / 2) ** 3)
                masas_g.append(volumen_cm3 * DENSIDAD_SOLIDO)

            # 5. Distribución
            df_particulas = pd.DataFrame({'Diametro_mm': diametros_mm, 'Masa_g': masas_g})
            if df_particulas.empty:
                st.error("No se pudieron detectar rocas. Intenta con una imagen de mayor contraste.")
                st.stop()
                
            masa_total = df_particulas['Masa_g'].sum()
            resultados_mallas, masa_acumulada = [], 0
            aberturas = list(mallas.values())
            nombres = list(mallas.keys())

            for i in range(len(aberturas)):
                if i == 0:
                    masa_retenida = df_particulas[df_particulas['Diametro_mm'] >= aberturas[i]]['Masa_g'].sum()
                else:
                    masa_retenida = df_particulas[(df_particulas['Diametro_mm'] < aberturas[i-1]) & 
                                                  (df_particulas['Diametro_mm'] >= aberturas[i])]['Masa_g'].sum()
                
                porcentaje_retenido = (masa_retenida / masa_total) * 100
                masa_acumulada += porcentaje_retenido
                resultados_mallas.append({
                    'Malla': nombres[i], 'Abertura_mm': aberturas[i], 
                    '% Retenido': porcentaje_retenido, '% Pasante': 100 - masa_acumulada
                })

            masa_finos = df_particulas[df_particulas['Diametro_mm'] < aberturas[-1]]['Masa_g'].sum()
            resultados_mallas.append({
                'Malla': 'Finos', 'Abertura_mm': 0,
                '% Retenido': (masa_finos / masa_total) * 100, '% Pasante': 0
            })
            
            df_dist = pd.DataFrame(resultados_mallas)
            f_interp = interp1d(df_dist['% Pasante'], df_dist['Abertura_mm'], fill_value="extrapolate")
            p80 = float(f_interp(80.0))

            # 6. Gráfico
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(df_dist['Abertura_mm'], df_dist['% Pasante'], 'o-', label='Curva Granulométrica')
            ax.axhline(y=80, color='r', linestyle='--', label='80% Pasante')
            ax.axvline(x=p80, color='g', linestyle='--', label=f'P80 = {p80:.2f} mm')
            ax.set_xscale('log')
            ax.set_xlabel('Abertura de Malla (mm)')
            ax.set_ylabel('% Pasante Acumulado')
            ax.set_title('Distribución Granulométrica')
            ax.grid(True, which="both", ls="-")
            ax.legend()
            
            st.pyplot(fig) 
            
            # ==========================================
            # 7. MEJORA: PDF EN UNA SOLA PÁGINA (A4)
            # ==========================================
            with tempfile.TemporaryDirectory() as tmpdirname:
                img_path = os.path.join(tmpdirname, "foto.jpg")
                plot_path = os.path.join(tmpdirname, "plot.png")
                pdf_path = os.path.join(tmpdirname, "reporte.pdf")
                
                # Guardar imagen original (rgb para que mantenga colores) y gráfico
                cv2.imwrite(img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                fig.savefig(plot_path, bbox_inches='tight')
                
                pdf = FPDF(orientation='P', unit='mm', format='A4')
                pdf.add_page()
                pdf.set_margins(10, 10, 10)
                
                # Título y P80
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(190, 10, txt="Reporte de Análisis Granulométrico", ln=True, align='C')
                pdf.set_font("Arial", 'B', 12)
                pdf.set_text_color(0, 100, 0)
                pdf.cell(190, 10, txt=f"P80 Calculado: {p80:.2f} mm", ln=True, align='C')
                
                # Imágenes lado a lado (Posición Y = 35)
                pdf.image(img_path, x=10, y=35, w=90)
                pdf.image(plot_path, x=105, y=35, w=95)
                
                # Bajar el cursor forzosamente debajo de las imágenes (aprox Y=130)
                pdf.set_y(130)
                
                # Tabla centrada y compacta
                pdf.set_font("Arial", 'B', 9)
                pdf.set_text_color(0, 0, 0)
                
                # Centrar tabla (ancho total 120mm. Margen izq = (210-120)/2 = 45)
                margen_izq = 45
                
                pdf.set_x(margen_izq)
                pdf.cell(40, 7, 'Malla', 1, 0, 'C')
                pdf.cell(40, 7, 'Abertura (mm)', 1, 0, 'C')
                pdf.cell(40, 7, '% Pasante', 1, 1, 'C') # El '1' al final hace el salto de línea
                
                pdf.set_font("Arial", '', 9)
                for index, row in df_dist.iterrows():
                    pdf.set_x(margen_izq)
                    pdf.cell(40, 6, str(row['Malla']), 1, 0, 'C')
                    pdf.cell(40, 6, f"{row['Abertura_mm']:.2f}", 1, 0, 'C')
                    pdf.cell(40, 6, f"{row['% Pasante']:.2f}%", 1, 1, 'C')
                
                pdf.output(pdf_path)
                
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
            
            st.success("¡Análisis completado con éxito!")
            st.download_button(
                label="📥 Descargar Reporte en PDF (1 Hoja)",
                data=pdf_bytes,
                file_name="Reporte_Granulometria_Mejorado.pdf",
                mime="application/pdf"
            )
