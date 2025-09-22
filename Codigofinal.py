# standard library
import collections
from enum import Enum
import math
import os.path
import pprint
import sys
import time

# third party library
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
import matplotlib.patheffects as path_effects

# Clase para colores de la terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

# Element namedtuple
Element = collections.namedtuple('Element', ['x0', 'y0', 'x1', 'y1', 'z'])

# MARGIN RATIO
MARGIN_RATIO = 0.95

# global variables
pp = pprint.PrettyPrinter(indent=4)

class GcodeType(Enum):
    """ enum of GcodeType """
    FDM_REGULAR = 1
    FDM_STRATASYS = 2
    LPBF_REGULAR = 3
    LPBF_SCODE = 4
    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)

class GcodeReader:
    """ Gcode reader class """
    def __init__(self, filename, filetype=GcodeType.FDM_REGULAR):
        if not os.path.exists(filename):
            print(f"{filename} does not exist!")
            sys.exit(1)
        self.filename = filename
        self.filetype = filetype
        self.n_segs = 0
        self.segs = None
        self.n_layers = 0
        self.seg_index = []
        self.xyzlimits = None
        self.minDimensions = None
        self.segs = self._read(filename)
        self.xyzlimits = self._compute_xyzlimits(self.segs)
        self.minDimensions = self.get_specimenDimensions()

    def _read(self, filename):
        if self.filetype == GcodeType.FDM_REGULAR:
            segs = self._read_fdm_regular(filename)
        else:
            print("file type is not supported")
            sys.exit(1)
        return segs
    
    def _compute_xyzlimits(self, seg_list):
        if len(seg_list) == 0:
            return (0, 1, 0, 1, 0, 1)
        seg_list_arr = np.array(seg_list)
        xmin = min(seg_list_arr[:, 0].min(), seg_list_arr[:, 2].min())
        xmax = max(seg_list_arr[:, 0].max(), seg_list_arr[:, 2].max())
        ymin = min(seg_list_arr[:, 1].min(), seg_list_arr[:, 3].min())
        ymax = max(seg_list_arr[:, 1].max(), seg_list_arr[:, 3].max())
        zmin = seg_list_arr[:, 4].min()
        zmax = seg_list_arr[:, 4].max()
        return (xmin, xmax, ymin, ymax, zmin, zmax)

    def _find_infill_pattern(self, filename):
        try:
            with open(filename, 'r', errors='ignore') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    clean_line = line.strip()
                    if not clean_line.startswith(';'):
                        continue
                    parameter_line = clean_line.lstrip('; ').strip()
                    parts = parameter_line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        if key == 'fill_pattern':
                            return value.capitalize()
        except Exception:
            return "No especificado"
        return "No especificado"

    def _read_fdm_regular(self, filename):
        with open(filename, 'r') as infile:
            lines = (line.strip() for line in infile.readlines() if line.strip())
            new_lines = []
            for line in lines:
                if line.startswith('G'):
                    idx = line.find(';')
                    if idx != -1:
                        line = line[:idx]
                    new_lines.append(line)
            lines = new_lines
       
        segs = []
        temp = -float('inf')
        gxyzef = [temp, temp, temp, temp, temp, temp, temp]
        d = dict(zip(['G', 'X', 'Y', 'Z', 'E', 'F', 'S'], range(7)))
        x0 = temp
        y0 = temp
        
        for line in lines:
            tokens = line.split()
            for token in tokens:
                if len(token) > 1 and token[0] in d:
                    gxyzef[d[token[0]]] = float(token[1:])

            if gxyzef[0] == 1:
                if np.isfinite(gxyzef[3]):
                   z = gxyzef[3]
                if np.isfinite(gxyzef[1]) and np.isfinite(gxyzef[2]) and not np.isfinite(gxyzef[4]):
                   x0 = gxyzef[1]
                   y0 = gxyzef[2]
                else:
                   if np.isfinite(gxyzef[1]) and np.isfinite(gxyzef[2]) and (gxyzef[4] > 0):
                      segs.append((x0, y0, gxyzef[1], gxyzef[2], z))
                      x0 = gxyzef[1]
                      y0 = gxyzef[2]
            gxyzef = [temp, temp, temp, temp, temp, temp, temp]
            
        segs = np.array(segs)
        segs[:, 4] = np.round(segs[:, 4], 2)
        
        self.n_segs = len(segs)
        self.seg_index = np.unique(segs[:, 4])
        self.n_layers = len(self.seg_index)
        infill_type = self._find_infill_pattern(filename)
        
        print(f"{Colors.CYAN}📏 Número de segmentos totales:{Colors.ENDC} {self.n_segs}")
        print(f"{Colors.CYAN}☰ Número de capas detectadas:{Colors.ENDC} {self.n_layers-1}")
        print(f"{Colors.CYAN}▦ Tipo de relleno:{Colors.ENDC} {infill_type}")
        
        return segs
   
    def get_specimenDimensions(self):
        n_zCoords = len(self.seg_index)
        if n_zCoords == 0: return [0, 0, 0, 0]
        mz = min(int(n_zCoords / 2), n_zCoords - 1)
        mz_idx = self.seg_index[mz]
        mz_layerSegs = self.get_layerSegs(mz_idx, mz_idx)
        if len(mz_layerSegs) == 0: return [0, 0, 0, 0]
        mz_layerSegs = np.array(mz_layerSegs)
        minx = min(np.min(mz_layerSegs[:, 0]), np.min(mz_layerSegs[:, 2]))
        miny = min(np.min(mz_layerSegs[:, 1]), np.min(mz_layerSegs[:, 3]))
        maxx = max(np.max(mz_layerSegs[:, 0]), np.max(mz_layerSegs[:, 2]))
        maxy = max(np.max(mz_layerSegs[:, 1]), np.max(mz_layerSegs[:, 3]))
        minDimensions = [minx, miny, maxx, maxy]
        return minDimensions

    def get_layerSegs(self, min_layer, max_layer):
        return [seg for seg in self.segs if min_layer <= seg[4] <= max_layer]

    def remove_skirt(self):
        new_segs = [seg for seg in self.segs if not self.is_skirt(seg)]
        self.segs = np.array(new_segs)
        self.xyzlimits = self._compute_xyzlimits(list(self.segs))
        self.seg_index = np.unique(self.segs[:, 4])
        self.n_layers = len(self.seg_index)
        print(f"{Colors.CYAN}📏 Número de segmentos de la probeta:{Colors.ENDC} {len(self.segs)}")

    def is_skirt(self, seg):
        minx, miny, maxx, maxy = self.minDimensions
        return not ((minx <= seg[0] <= maxx and miny <= seg[1] <= maxy) and
                    (minx <= seg[2] <= maxx and miny <= seg[3] <= maxy))

    def search_minorArea(self, delta, step, ejeMenor, ejeMayor):
        minx = self.minDimensions[0]
        maxx = self.minDimensions[2]
        middleP = minx + ((maxx - minx) / 2)
        limInf = middleP - (delta / 2)
        limSup = middleP + (delta / 2)
        ptoCortes = np.arange(limInf, limSup, step)
        minArea = np.inf
        minP = 0
        
        for p in ptoCortes:
            (areaP, _, _, _) = self.apply_cutPoint(p, ejeMenor, ejeMayor)
            if areaP < minArea:
                minArea = areaP
                minP = p
        
        (final_area, n_cut, solid_area, final_cut_points) = self.apply_cutPoint(minP, ejeMenor, ejeMayor)
        
        print("\n" + Colors.GREEN + "="*50 + Colors.ENDC)
        print(f"{Colors.BOLD}{Colors.GREEN}    🏆 RESULTADOS DEL ANÁLISIS DE CORTE 🏆{Colors.ENDC}")
        print(Colors.GREEN + "="*50 + Colors.ENDC)
        print(f"{Colors.BLUE}✔️ Menor área proporcional encontrada:{Colors.ENDC} {Colors.YELLOW}{final_area:.6f}{Colors.ENDC}")
        print(f"{Colors.BLUE}📍 Ubicación del corte óptimo (eje X):{Colors.ENDC} {Colors.YELLOW}{minP:.6f} mm{Colors.ENDC}")
        print(f"{Colors.BLUE}✂️ Puntos de corte encontrados:{Colors.ENDC} {Colors.YELLOW}{n_cut}{Colors.ENDC}")
        print(f"{Colors.BLUE}🧱 Área sólida total del corte:{Colors.ENDC} {Colors.YELLOW}{solid_area:.12f}{Colors.ENDC}")
        
        self.animate_layer_dynamic(final_cut_points, [minP, self.minDimensions[1], self.minDimensions[3]])
        self.animate_layers2(final_cut_points, [minP, self.minDimensions[1], self.minDimensions[3]])
        self.figSolidRectangle(final_cut_points)
        
        return (minArea, minP)
        
    def apply_cutPoint(self, xcorte, ejeMenor, ejeMayor):
        miny  = self.minDimensions[1]
        maxy  = self.minDimensions[3]
        cutSeg = [xcorte, miny, maxy]
        cutPoints = self.apply_cutSeg(cutSeg)
        if not cutPoints:
            return (float('inf'), 0, 0, [])
        extremePoints = self.elispse_extremePoints(cutPoints, ejeMenor, ejeMayor)
        area_totalSolida = self.calcular_areaTotal_solida(extremePoints)
        minDist_y = 0.33799999999999386
        areaP = self.estimate_proportionalArea(cutPoints, area_totalSolida, minDist_y)
        return (areaP, len(cutPoints), area_totalSolida, cutPoints)

    def apply_cutSeg(self, cutSeg):
        cutPoints = []
        xcorte = cutSeg[0]
        for (x0, y0, x1, y1, z) in self.segs:
            if x0 == x1:
                if (xcorte >= min(x0, x1)) and (xcorte <= max(x0, x1)):
                    cutPoints.append([xcorte, y0, z, x0, x1, y0, y0])
            else:
                if min(x0, x1) <= xcorte and max(x0, x1) >= xcorte:
                    mseg = (y1-y0) / (x1-x0)
                    y = mseg * (xcorte - x0) + y0
                    if (y >= cutSeg[1]) and (y <= cutSeg[2]):
                        cutPoints.append([xcorte, y, z, x0, x1, y0, y1])
        return cutPoints 

    def estimate_proportionalArea(self, cutPoints, areaSolida, minDist_y):
        if not cutPoints: return 0
        y_coords = [point[1] for point in cutPoints]
        z_coords = [point[2] for point in cutPoints]
        miny = min(y_coords); maxy = max(y_coords)
        nPoints_y = round((maxy-miny)/minDist_y)
        nPoints_z = len(np.unique(z_coords))
        nCutPoints = len(cutPoints)
        if nPoints_y * nPoints_z == 0: return 0
        nGridPoints = nPoints_y * nPoints_z
        areaEstimada = (areaSolida * nCutPoints) / nGridPoints
        return min(areaEstimada, areaSolida)

    def elispse_extremePoints(self, cutPoints, ejeMenor, ejeMayor):
        if not cutPoints: return []
        extremePoints = []
        for point in cutPoints:
            x, y, z = point[0], point[1], point[2]
            extremePoints.extend([
                [x, y, z + ejeMenor], [x, y, z - ejeMenor],
                [x, y + ejeMayor, z], [x, y - ejeMayor, z]
            ])
        return extremePoints

    # --- INICIO MODIFICACIÓN: Función de título mejorada ---
    def draw_neon_title(self, fig, ax, title, color):
        # El 'color' es el color del brillo, el texto principal será blanco.
        title_text = ax.set_title(title, fontsize=16, fontweight='bold', color='white', y=1.02)
        # Se aplica un "contorno" de color para simular el brillo.
        title_text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=color, alpha=0.8),
            path_effects.Normal()
        ])
        fig.canvas.draw()
    # --- FIN MODIFICACIÓN ---

    def animate_layer_dynamic(self, cutPoints, cutSeg):
        minx, miny, maxx, maxy = self.minDimensions
        margin_x = (maxx - minx) * 0.05
        margin_y = (maxy - miny) * 0.05
        fig, ax = plt.subplots(figsize=(12, 8))

        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        ax.set_xlabel('X [mm]', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Y [mm]', fontsize=12, fontweight='bold', color='white')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3, color='gray')
        ax.tick_params(colors='white', which='both')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=9, verticalalignment='top', 
                            color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.8, edgecolor='none'))
        
        ax.set_xlim(minx - margin_x, maxx + margin_x)
        ax.set_ylim(miny - margin_y, maxy + margin_y)
        ax.set_aspect('equal', adjustable='box')
        
        tracer, = ax.plot([], [], 'o', color='#08F7FE', markersize=6, zorder=10)
        
        plt.show(block=False)
        self.draw_neon_title(fig, ax, "Análisis de Patrón de Relleno 2D", color='#FFD700')
        
        probeta_segs = [seg for seg in self.segs if (minx <= seg[0] <= maxx and miny <= seg[1] <= maxy and minx <= seg[2] <= maxx and miny <= seg[3] <= maxy)]
        unique_layers = sorted(list(set([seg[4] for seg in probeta_segs])))
        selected_layers = []
        num_unique_layers = len(unique_layers)
        
        if num_unique_layers >= 5:
            layer_40_percent_index = int(num_unique_layers * 0.4)
            layer_60_percent_index = int(num_unique_layers * 0.6)
            selected_layers.append(unique_layers[layer_40_percent_index])
            if layer_40_percent_index != layer_60_percent_index:
                selected_layers.append(unique_layers[layer_60_percent_index])
        elif num_unique_layers > 2:
            selected_layers.append(unique_layers[num_unique_layers // 2])
            
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        segment_counter = 0
        
        for layer_idx, z_layer in enumerate(selected_layers):
            layer_segs = [seg for seg in probeta_segs if seg[4] == z_layer]
            color = colors[layer_idx % len(colors)]
            
            for (x0, y0, x1, y1, z) in layer_segs:
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=0.7, alpha=0.8, solid_capstyle='round')
                tracer.set_data([x1], [y1])
                segment_counter += 1
                info_text.set_text(f'Capa {layer_idx+1}/{len(selected_layers)}\nZ={z:.1f}mm\nSegs: {segment_counter}')
                plt.pause(0.01)

        info_text.set_text(f'PATRÓN COMPLETADO\n{len(selected_layers)} capas mostradas\nAnalizando zona crítica...')
        tracer.set_visible(False)
        plt.pause(0.5)

        if cutSeg:
            info_text.set_text(f'ANÁLISIS DE CORTE\nX = {cutSeg[0]:.2f} mm\nIdentificando zona crítica')
            for alpha in np.linspace(0, 1, 10):
                if hasattr(ax, '_cut_line'):
                    ax._cut_line.remove()
                ax._cut_line = ax.plot([cutSeg[0], cutSeg[0]], [cutSeg[1], cutSeg[2]], color='#FF4F4F', linewidth=3, alpha=alpha)[0]
                plt.pause(0.03)

        if cutPoints:
            step_points = max(1, len(cutPoints) // 15)
            filtered_points = cutPoints[::step_points]
            plt.pause(0.3)
            info_text.set_text(f'PUNTOS CRÍTICOS\n{len(filtered_points)} de {len(cutPoints)} puntos\nZona de menor resistencia')
            cut_x_filtered = [p[0] for p in filtered_points]
            cut_y_filtered = [p[1] for p in filtered_points]
            scatter_points = ax.scatter(cut_x_filtered, cut_y_filtered, color='#FF4F4F', s=50, marker='o', edgecolors='white', linewidth=1.5, alpha=0.9, zorder=15)
            for _ in range(2):
                scatter_points.set_sizes([80] * len(filtered_points))
                plt.pause(0.2)
                scatter_points.set_sizes([50] * len(filtered_points))
                plt.pause(0.2)
        
        self.draw_neon_title(fig, ax, "Zona Crítica Identificada", color='#FFD700')
        
        red_patch = mpatches.Patch(color='#FF4F4F', label=f'Corte / Puntos Críticos ({len(filtered_points)})')
        blue_patch = mpatches.Patch(color='#2E86AB', label='Recorrido de Relleno')
        legend = ax.legend(handles=[red_patch, blue_patch], facecolor='#333333', edgecolor='white')
        plt.setp(legend.get_texts(), color='white')
        
        explanation_text = "Este gráfico muestra una simulación del patrón de relleno en capas representativas de la probeta.\nLa línea roja y los puntos críticos resaltan la zona de menor resistencia estructural identificada."
        fig.text(0.5, 0.01, explanation_text, ha='center', va='bottom', fontsize=9, style='italic', color='white')
        fig.subplots_adjust(bottom=0.15)
        
        print(f"\n{Colors.BLUE}📊 Mostrando Gráfico 1: {Colors.BOLD}Análisis de Patrón de Relleno 2D{Colors.ENDC}")
        print(f"{Colors.YELLOW}   Cierre esta ventana para continuar...{Colors.ENDC}")
        plt.show()
        print(f"{Colors.GREEN}✅ Ventana del Gráfico 1 cerrada. Preparando el siguiente gráfico...{Colors.ENDC}")

    def animate_layers2(self, cutPoints, cutSeg):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        
        ax.xaxis.pane.set_edgecolor('#555555')
        ax.yaxis.pane.set_edgecolor('#555555')
        ax.zaxis.pane.set_edgecolor('#555555')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        layers_to_show_z = self.seg_index
        plot_segs = [seg for seg in self.segs if seg[4] in layers_to_show_z]
        if not plot_segs: return
        plot_xmin, plot_xmax, plot_ymin, plot_ymax, plot_zmin, plot_zmax = self._compute_xyzlimits(plot_segs)
        z_scale_factor = 120.0
        x_range = plot_xmax - plot_xmin
        y_range = plot_ymax - plot_ymin
        x_mid = plot_xmin + x_range / 2
        y_mid = plot_ymin + y_range / 2
        max_xy_range = max(x_range, y_range)
        plot_radius_xy = max_xy_range * 0.5 * 1.05 
        ax.set_xlim(x_mid - plot_radius_xy, x_mid + plot_radius_xy)
        ax.set_ylim(y_mid - plot_radius_xy, y_mid + plot_radius_xy)
        scaled_zmax = plot_zmin + (plot_zmax - plot_zmin) * z_scale_factor
        z_range_scaled = scaled_zmax - plot_zmin
        ax.set_zlim(plot_zmin - z_range_scaled * 0.05, scaled_zmax + z_range_scaled * 0.05)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        
        ax.set_box_aspect((1, 1, 1.5)) 

        plt.show(block=False)
        self.draw_neon_title(fig, ax, "Probeta 3D con Plano de Corte", color='#FF00FF')
        
        layer_colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(layers_to_show_z)))
        segment_counter = 0
        for idx, z_layer in enumerate(layers_to_show_z):
            layer_segs = self.get_layerSegs(z_layer, z_layer)
            color = layer_colors[idx]
            for (x0, y0, x1, y1, z) in layer_segs:
                plot_z = plot_zmin + (z - plot_zmin) * z_scale_factor
                ax.plot([x0, x1], [y0, y1], [plot_z, plot_z], color=color, linewidth=2, alpha=0.7)
                segment_counter += 1
                if segment_counter % 10 == 0:
                    plt.pause(0.00001)

        if cutPoints:
            filtered_cut_points = [p for p in cutPoints if p[2] in layers_to_show_z]
            if filtered_cut_points:
                x_coords_filtered = [p[0] for p in filtered_cut_points]
                y_coords_filtered = [p[1] for p in filtered_cut_points]
                
                if len(layers_to_show_z) > 0:
                    z_coords_filtered = [plot_zmin + (p[2] - plot_zmin) * z_scale_factor for p in filtered_cut_points]
                    ax.scatter(x_coords_filtered, y_coords_filtered, z_coords_filtered, color='#4FFF4F', s=25, marker='o', zorder=10, label=f'Puntos Críticos ({len(filtered_cut_points)})')

                if cutSeg:
                    xcorte = cutSeg[0]
                    min_y_cut = min(y_coords_filtered)
                    max_y_cut = max(y_coords_filtered)
                    
                    verts = [[(xcorte, min_y_cut, plot_zmin),
                              (xcorte, max_y_cut, plot_zmin),
                              (xcorte, max_y_cut, scaled_zmax),
                              (xcorte, min_y_cut, scaled_zmax)]]
                    plane = Poly3DCollection(verts, alpha=0.2, facecolor='#FF4F4F', edgecolor=None)
                    ax.add_collection3d(plane)
                
                handles, labels = ax.get_legend_handles_labels()
                if cutSeg:
                    plane_legend = mpatches.Patch(color='#FF4F4F', alpha=0.4, label=f'Plano de Corte (X={cutSeg[0]:.2f})')
                    handles.append(plane_legend)
                
                legend = ax.legend(handles=handles, facecolor='#333333', edgecolor='white')
                plt.setp(legend.get_texts(), color='white')
        
        explanation_text = "Esta vista 3D muestra la estructura interna de la probeta completa. El plano rojo indica la ubicación\ndel corte, y los puntos verdes marcan los puntos críticos a lo largo de las diferentes capas."
        fig.text(0.5, 0.05, explanation_text, ha='center', va='bottom', fontsize=9, style='italic', color='white')

        print(f"\n{Colors.BLUE}🧊 Mostrando Gráfico 2: {Colors.BOLD}Probeta 3D con Plano de Corte{Colors.ENDC}")
        print(f"{Colors.YELLOW}   Cierre esta ventana para continuar...{Colors.ENDC}")
        plt.show()
        print(f"{Colors.GREEN}✅ Ventana del Gráfico 2 cerrada. Preparando el último gráfico...{Colors.ENDC}")

    def figSolidRectangle(self, cutPoints):
        if not cutPoints:
            print(f"{Colors.YELLOW}Advertencia: No hay puntos de corte para graficar el Gráfico 3.{Colors.ENDC}")
            return

        y_coords = [p[1] for p in cutPoints]
        z_coords = [p[2] for p in cutPoints]
        
        miny, maxy = min(y_coords), max(y_coords)
        minz, maxz = min(z_coords), max(z_coords)
        
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        ax.set_xlabel("Y (mm)", color='white')
        ax.set_ylabel("Z (mm)", color='white')
        
        ax.grid(True, linestyle='--', alpha=0.2, color='gray')
        ax.tick_params(colors='white', which='both')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        ax.set_xlim(miny - 1, maxy + 1)
        ax.set_ylim(minz - 1, maxz + 1)
        ax.set_aspect('equal', adjustable='box')
        
        plt.show(block=False)
        self.draw_neon_title(fig, ax, "Análisis de Zona Crítica", color='#39FF14')
        
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=9, 
                            verticalalignment='top', horizontalalignment='left', color='cyan')
        
        material_points = np.array([[p[1], p[2]] for p in cutPoints])
        
        info_text.set_text('Analizando puntos de material...')
        
        point_collection = ax.scatter(material_points[:, 0], material_points[:, 1], s=20, color='#FF4F4F', alpha=0)
        plt.pause(0.5)

        alphas = np.zeros(len(material_points))
        batch_size = 50
        for i in range(0, len(material_points), batch_size):
            alphas[i:i+batch_size] = 0.8
            point_collection.set_alpha(alphas)
            plt.pause(0.1)

        info_text.set_text('Resaltando zona de material...')
        plt.pause(0.5)
        for _ in range(2):
            point_collection.set_sizes([40] * len(material_points))
            plt.pause(0.4)
            point_collection.set_sizes([20] * len(material_points))
            plt.pause(0.4)

        info_text.set_text('Detectando zonas vacías...')
        plt.pause(0.5)

        try:
            from scipy.spatial import ConvexHull
            from sklearn.cluster import DBSCAN
            from matplotlib.path import Path
            
            if len(material_points) < 3:
                red_patch = mpatches.Patch(color='#FF4F4F', label=f'Material Presente ({len(cutPoints)} puntos)')
                legend = ax.legend(handles=[red_patch], facecolor='#333333', edgecolor='white')
                plt.setp(legend.get_texts(), color='white')
                print(f"\n{Colors.BLUE}📉 Mostrando Gráfico 3: {Colors.BOLD}Vista Frontal de Zonas Críticas{Colors.ENDC}")
                print(f"{Colors.YELLOW}   Cierre esta ventana para finalizar el análisis...{Colors.ENDC}")
                plt.show()
                print(f"\n{Colors.GREEN}✅ Ventana del Gráfico 3 cerrada.{Colors.ENDC}")
                print(f"\n{Colors.HEADER}{Colors.BOLD}🎉 Todos los gráficos han sido mostrados. El análisis ha finalizado.{Colors.ENDC}\n")
                return 
            
            outer_hull = ConvexHull(material_points)
            outer_path = Path(material_points[outer_hull.vertices])
            grid_resolution = 50
            grid_y, grid_z = np.meshgrid(np.linspace(miny, maxy, grid_resolution), np.linspace(minz, maxz, grid_resolution))
            grid_points = np.vstack([grid_y.ravel(), grid_z.ravel()]).T
            inside_mask = outer_path.contains_points(grid_points)
            points_inside_material = grid_points[inside_mask]
            void_candidates = []
            tolerance = 0.25 
            for point in points_inside_material:
                distances = np.linalg.norm(material_points - point, axis=1)
                if np.min(distances) > tolerance:
                    void_candidates.append(point)

            if not void_candidates:
                info_text.set_text('No se detectaron vacíos significativos.')
            else:
                info_text.set_text('Delimitando contornos internos...')
                plt.pause(0.5)
                void_points = np.array(void_candidates)
                grid_spacing = (maxy - miny) / grid_resolution
                clustering = DBSCAN(eps=grid_spacing * 2.0, min_samples=4).fit(void_points)
                labels = clustering.labels_
                unique_labels = set(labels)
                
                for k in unique_labels:
                    if k == -1 or len(void_points[labels == k]) < 3:
                        continue
                    cluster_points = void_points[labels == k]
                    hull = ConvexHull(cluster_points)
                    for simplex in hull.simplices:
                        ax.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], color='#4FFF4F', linestyle='--', linewidth=3.5, alpha=0.4)
                        ax.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], color='#4FFF4F', linestyle='--', linewidth=2.0, alpha=1)
                        plt.pause(0.2)
                
                info_text.set_text('Análisis completado.')
            
            red_patch = mpatches.Patch(color='#FF4F4F', label=f'Material Presente ({len(cutPoints)} puntos)')
            green_patch = mpatches.Patch(color='#4FFF4F', label='Contorno Interno (Menos Material)')
            legend = ax.legend(handles=[red_patch, green_patch], facecolor='#333333', edgecolor='white')
            plt.setp(legend.get_texts(), color='white')

        except ImportError:
             print(f"\n{Colors.RED}Error: Faltan librerías cruciales para el Gráfico 3.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}Ocurrió un error al generar los polígonos de vacíos: {e}{Colors.ENDC}")
        
        explanation_text = "Esta es una vista frontal de la sección de corte. Los puntos rojos representan la presencia de material y\nlos contornos verdes delimitan las áreas con menor densidad (vacíos), que son las más propensas a fallar."
        fig.text(0.5, 0.01, explanation_text, ha='center', va='bottom', fontsize=9, style='italic', color='white')
        fig.subplots_adjust(bottom=0.15)
        
        print(f"\n{Colors.BLUE}📉 Mostrando Gráfico 3: {Colors.BOLD}Vista Frontal de Zonas Críticas{Colors.ENDC}")
        print(f"{Colors.YELLOW}   Cierre esta ventana para finalizar el análisis...{Colors.ENDC}")
        plt.show()
        print(f"\n{Colors.GREEN}✅ Ventana del Gráfico 3 cerrada.{Colors.ENDC}")
        print(f"\n{Colors.HEADER}{Colors.BOLD}🎉 Todos los gráficos han sido mostrados. El análisis ha finalizado.{Colors.ENDC}\n")

    def calcular_areaTotal_solida(self, extremePoints):
        if not extremePoints: return 0
        y_coords = [point[1] for point in extremePoints]
        miny = min(y_coords); maxy = max(y_coords)
        minz = min(self.segs[:, 4]); maxz = max(self.segs[:, 4])
        a = maxy - miny
        b = maxz - minz
        return a*b

    def min_coordxy(self, segList):
        segArray = np.array(segList)
        minx = min(np.min(segArray[:, 0]), np.min(segArray[:, 2]))
        miny = min(np.min(segArray[:, 1]), np.min(segArray[:, 3]))
        return (minx, miny)
                 
    def max_coordxy(self, segList):
        segArray = np.array(segList)
        maxx = max(np.max(segArray[:, 0]), np.max(segArray[:, 2]))
        maxy = max(np.max(segArray[:, 1]), np.max(segArray[:, 3]))
        return (maxx, maxy)

def create_axis(figsize=(8, 8), projection='2d'):
    projection = projection.lower()
    if projection not in ['2d', '3d']: raise ValueError
    if projection == '2d':
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    return fig, ax

def command_line_runner(filename, filetype):
    delta = 7.62; step = 0.1
    ejeMenor = 0.119 * 2; ejeMayor = 0.191 * 2
    
    print("\n" + Colors.HEADER + "="*50 + Colors.ENDC)
    print(f"{Colors.BOLD}{Colors.HEADER}    🤖 ANÁLISIS DE PROBETA G-CODE (Versión Final) 🤖{Colors.ENDC}")
    print(Colors.HEADER + "="*50 + Colors.ENDC)
    print(f"\n{Colors.BOLD}📂 Archivo de entrada:{Colors.ENDC} {Colors.CYAN}{filename}{Colors.ENDC}")
    
    gcode_reader = GcodeReader(filename, filetype)
    
    print(f"\n{Colors.BOLD}⚙️  Procesando y limpiando datos...{Colors.ENDC}")
    gcode_reader.remove_skirt()
    time.sleep(0.5)
    
    print(f"\n{Colors.BOLD}🔎 Iniciando búsqueda de área mínima...{Colors.ENDC}")
    time.sleep(0.5)
    gcode_reader.search_minorArea(delta, step, ejeMenor, ejeMayor)

if __name__ == "__main__":
    filetype = GcodeType.FDM_REGULAR
    if len(sys.argv) < 2:
        print(f"{Colors.RED}Error: Proporciona el nombre del archivo G-code como argumento.{Colors.ENDC}")
        sys.exit(1)
    filename = sys.argv[1]
    command_line_runner(filename, filetype)