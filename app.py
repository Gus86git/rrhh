import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importaciones para gráficos avanzados
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Configuración de página
st.set_page_config(
    page_title="RRHH Analytics Pro",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados mejorados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .alert-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #ff4757;
    }
    .alert-medium {
        background: linear-gradient(135deg, #ffd93d 0%, #ffcd3c 100%);
        color: black;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #ff9f1a;
    }
    .alert-low {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #219a52;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, #3498db, transparent);
        padding-left: 1rem;
    }
    .employee-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3498db;
    }
    .apto-card {
        border-left: 4px solid #2ecc71 !important;
    }
    .no-apto-card {
        border-left: 4px solid #e74c3c !important;
    }
    .manual-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Generar datos sintéticos completos para la demo"""
    np.random.seed(42)
    
    # Generar empleados con criterios de aptitud
    nombres = ['Sofia', 'Martina', 'Lucia', 'Ana', 'Carolina', 'Valentina', 
               'Carlos', 'Diego', 'Juan', 'Pablo', 'Ricardo', 'Javier', 
               'Miguel', 'Roberto', 'Fernando', 'Laura', 'Gabriela', 'Mariana']
    apellidos = ['Lopez', 'Gonzalez', 'Garcia', 'Martinez', 'Rodriguez', 
                 'Perez', 'Diaz', 'Gomez', 'Fernandez', 'Romero', 'Silva', 'Torres']
    
    # Especialidades por departamento
    especialidades = {
        'Albañilería': ['Albañil Maestro', 'Ayudante Albañil', 'Enfoscador', 'Colocador Cerámico'],
        'Electricidad': ['Electricista Industrial', 'Electricista Residencial', 'Técnico Electrónico'],
        'Plomería': ['Instalador Sanitario', 'Gasista Matriculado', 'Técnico HVAC'],
        'Herrería': ['Soldador Especializado', 'Herrero Estructural', 'Calderero'],
        'Pintura': ['Pintor Industrial', 'Pintor Decorativo', 'Aplicador Especializado']
    }
    
    certificaciones = {
        'Albañilería': ['Hormigón Armado', 'Encofrados', 'Seguridad en Altura'],
        'Electricidad': ['AT1', 'BT', 'Instalaciones MT', 'Automatización'],
        'Plomería': ['Gasista Matriculado', 'Termofusión', 'Sistemas HVAC'],
        'Herrería': ['Soldadura TIG', 'Soldadura MIG', 'Estructuras Metálicas'],
        'Pintura': ['Pintura Epoxi', 'Anticorrosivos', 'Texturas']
    }
    
    empleados = []
    for i in range(200):
        genero = np.random.choice(['Femenino', 'Masculino'], p=[0.35, 0.65])
        dept = np.random.choice(['Albañilería', 'Electricidad', 'Plomería', 'Herrería', 'Pintura'], 
                               p=[0.3, 0.25, 0.2, 0.15, 0.1])
        
        # Salarios base por departamento
        salario_base = {
            'Albañilería': 80000, 'Electricidad': 95000, 
            'Plomería': 85000, 'Herrería': 110000, 'Pintura': 75000
        }[dept]
        
        salario = salario_base * np.random.uniform(0.8, 1.5)
        experiencia = np.random.randint(6, 180)
        edad = np.random.randint(22, 60)
        
        # Determinar aptitud para obra compleja
        apto_obra_compleja = (
            (experiencia > 24) and 
            (np.random.random() > 0.3) and
            (edad >= 25 and edad <= 55)
        )
        
        # Certificaciones
        certs_disponibles = certificaciones[dept]
        num_certs = np.random.randint(1, min(4, len(certs_disponibles) + 1))
        certificaciones_empleado = np.random.choice(certs_disponibles, num_certs, replace=False)
        
        empleados.append({
            'id': f"EMP{i+1:03d}",
            'nombre': np.random.choice(nombres),
            'apellido': np.random.choice(apellidos),
            'genero': genero,
            'edad': edad,
            'departamento': dept,
            'especialidad': np.random.choice(especialidades[dept]),
            'cargo': f"{dept} {'Senior' if experiencia > 60 else 'Junior' if experiencia > 24 else 'Aprendiz'}",
            'salario': round(salario, 2),
            'fecha_contratacion': datetime.now() - timedelta(days=np.random.randint(30, 365*5)),
            'experiencia_meses': experiencia,
            'ubicacion': np.random.choice(['Sede Central', 'Obra Norte', 'Obra Sur', 'Obra Este', 'Obra Oeste']),
            'nivel_educacion': np.random.choice(['Secundario', 'Terciario', 'Universitario', 'Maestría'], 
                                              p=[0.4, 0.3, 0.2, 0.1]),
            'certificaciones': ', '.join(certificaciones_empleado),
            'apto_obra_compleja': apto_obra_compleja,
            'disponible_viaje': np.random.choice([True, False], p=[0.7, 0.3]),
            'vehiculo_propio': np.random.choice([True, False], p=[0.6, 0.4]),
            'activo': np.random.choice([True, False], p=[0.92, 0.08]),
            'evaluacion_desempeno': np.random.normal(85, 10),
            'ausencias_ultimo_mes': np.random.poisson(1.5)
        })
    
    df_empleados = pd.DataFrame(empleados)
    df_empleados['evaluacion_desempeno'] = df_empleados['evaluacion_desempeno'].clip(50, 100)
    
    # Generar obras con requisitos específicos
    obras = []
    tipos_obra = ['Residencial', 'Comercial', 'Industrial', 'Infraestructura', 'Institucional']
    
    for i in range(15):
        tipo_obra = np.random.choice(tipos_obra)
        complejidad = np.random.choice(['Baja', 'Media', 'Alta'], p=[0.3, 0.5, 0.2])
        
        # Requisitos basados en tipo y complejidad
        requisitos = {
            'Residencial': {'apto_obra_compleja': False, 'exp_minima': 12},
            'Comercial': {'apto_obra_compleja': complejidad != 'Baja', 'exp_minima': 24},
            'Industrial': {'apto_obra_compleja': True, 'exp_minima': 36},
            'Infraestructura': {'apto_obra_compleja': True, 'exp_minima': 48},
            'Institucional': {'apto_obra_compleja': complejidad == 'Alta', 'exp_minima': 24}
        }[tipo_obra]
        
        obras.append({
            'id': f"OBR{i+1:03d}",
            'nombre': f"Proyecto {tipo_obra} {i+1}",
            'tipo': tipo_obra,
            'ubicacion': np.random.choice(['Nordelta', 'Pilar', 'Tigre', 'Escobar', 'San Isidro', 'Belgrano', 'Palermo']),
            'presupuesto': np.random.randint(5000000, 30000000),
            'fecha_inicio': datetime.now() - timedelta(days=np.random.randint(30, 400)),
            'duracion_estimada': np.random.randint(90, 540),
            'estado': np.random.choice(['En Planificación', 'En Progreso', 'En Riesgo', 'Completado', 'Pausado'], 
                                     p=[0.1, 0.5, 0.15, 0.1, 0.15]),
            'gerente': np.random.choice([f"{emp['nombre']} {emp['apellido']}" for emp in empleados[:25]]),
            'complejidad': complejidad,
            'requiere_apto_obra_compleja': requisitos['apto_obra_compleja'],
            'experiencia_minima_meses': requisitos['exp_minima'],
            'requiere_vehiculo': np.random.choice([True, False], p=[0.6, 0.4]),
            'zona_riesgo': np.random.choice([True, False], p=[0.3, 0.7])
        })
    
    df_obras = pd.DataFrame(obras)
    
    # Generar asistencias y rendimiento
    asistencias = []
    for _ in range(3000):
        emp_idx = np.random.randint(0, len(empleados))
        emp = empleados[emp_idx]
        obra_idx = np.random.randint(0, len(obras))
        obra = obras[obra_idx]
        
        fecha = datetime.now() - timedelta(days=np.random.randint(1, 180))
        
        # Calcular productividad basada en aptitud y experiencia
        productividad_base = np.random.normal(85, 10)
        if emp['apto_obra_compleja'] and obra['requiere_apto_obra_compleja']:
            productividad_base += 5
        if emp['experiencia_meses'] >= obra['experiencia_minima_meses']:
            productividad_base += 3
        
        asistencias.append({
            'empleado_id': emp['id'],
            'obra_id': obra['id'],
            'fecha': fecha,
            'horas_trabajadas': np.random.randint(6, 10),
            'horas_extra': np.random.choice([0, 0, 0, 1, 2, 3], p=[0.4, 0.2, 0.15, 0.15, 0.07, 0.03]),
            'productividad': productividad_base,
            'calidad_trabajo': np.random.normal(90, 5),
            'incidentes_seguridad': np.random.poisson(0.05),
            'ausente': np.random.choice([True, False], p=[0.03, 0.97])
        })
    
    df_asistencias = pd.DataFrame(asistencias)
    df_asistencias['productividad'] = df_asistencias['productividad'].clip(50, 100)
    df_asistencias['calidad_trabajo'] = df_asistencias['calidad_trabajo'].clip(70, 100)
    
    return df_empleados, df_obras, df_asistencias

def create_advanced_plotly_chart(data, title, chart_type='bar', **kwargs):
    """Función avanzada para crear gráficos Plotly con estilo Power BI"""
    try:
        if chart_type == 'sunburst':
            fig = px.sunburst(data, **kwargs)
        elif chart_type == 'treemap':
            fig = px.treemap(data, **kwargs)
        elif chart_type == 'violin':
            fig = px.violin(data, **kwargs)
        elif chart_type == 'density_heatmap':
            fig = px.density_heatmap(data, **kwargs)
        elif chart_type == 'parallel_categories':
            fig = px.parallel_categories(data, **kwargs)
        elif chart_type == 'funnel':
            fig = px.funnel(data, **kwargs)
        elif chart_type == 'waterfall':
            fig = go.Figure(go.Waterfall(**kwargs))
        elif chart_type == 'indicator':
            fig = go.Figure(go.Indicator(**kwargs))
        else:
            # Usar plotly express para tipos básicos
            if chart_type == 'bar':
                fig = px.bar(data, **kwargs)
            elif chart_type == 'pie':
                fig = px.pie(data, **kwargs)
            elif chart_type == 'scatter':
                fig = px.scatter(data, **kwargs)
            elif chart_type == 'line':
                fig = px.line(data, **kwargs)
            elif chart_type == 'histogram':
                fig = px.histogram(data, **kwargs)
            elif chart_type == 'box':
                fig = px.box(data, **kwargs)
            else:
                fig = px.bar(data, **kwargs)
        
        # Estilo Power BI
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20, color='#2c3e50')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creando gráfico {title}: {str(e)}")
        return None

def main():
    # Header principal
    st.markdown('<h1 class="main-header">🏗️ RRHH Analytics Pro</h1>', unsafe_allow_html=True)
    
    # Cargar datos
    df_empleados, df_obras, df_asistencias = load_data()
    
    # Sidebar - Navegación
    st.sidebar.title("🏢 RRHH Analytics Pro")
    st.sidebar.markdown("---")
    
    menu = st.sidebar.radio(
        "📋 Módulos:",
        ["📊 Dashboard Ejecutivo", "👥 Gestión de Personal", "🏗️ Gestión de Obras", 
         "🎯 Aptitud para Obras", "📈 Analytics Avanzado", "⚠️ Alertas", 
         "📖 Manual del Dashboard", "⚙️ Configuración"]
    )
    
    # KPIs Principales - Siempre visibles
    st.markdown("### 📈 Métricas Clave en Tiempo Real")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_empleados = len(df_empleados[df_empleados['activo']])
        st.metric("👥 Empleados Activos", total_empleados, delta="+5%")
    
    with col2:
        aptos_obra_compleja = len(df_empleados[(df_empleados['activo']) & (df_empleados['apto_obra_compleja'])])
        st.metric("✅ Aptos Obra Compleja", aptos_obra_compleja, delta="+8%")
    
    with col3:
        productividad_promedio = df_asistencias['productividad'].mean()
        st.metric("📊 Productividad", f"{productividad_promedio:.1f}%", delta="+2.1%")
    
    with col4:
        rotacion = len(df_empleados[~df_empleados['activo']]) / len(df_empleados) * 100
        st.metric("🔄 Rotación", f"{rotacion:.1f}%", delta="-1.2%", delta_color="inverse")
    
    with col5:
        obras_activas = len(df_obras[df_obras['estado'] == 'En Progreso'])
        st.metric("🏗️ Obras Activas", obras_activas)
    
    st.markdown("---")
    
    # Contenido según menú seleccionado
    if menu == "📊 Dashboard Ejecutivo":
        show_executive_dashboard(df_empleados, df_obras, df_asistencias)
    elif menu == "👥 Gestión de Personal":
        show_person_management(df_empleados, df_asistencias)
    elif menu == "🏗️ Gestión de Obras":
        show_project_management(df_obras, df_asistencias, df_empleados)
    elif menu == "🎯 Aptitud para Obras":
        show_aptitude_analysis(df_empleados, df_obras)
    elif menu == "📈 Analytics Avanzado":
        show_advanced_analytics(df_empleados, df_asistencias)
    elif menu == "⚠️ Alertas":
        show_early_warnings(df_empleados, df_obras, df_asistencias)
    elif menu == "📖 Manual del Dashboard":
        show_dashboard_manual()
    elif menu == "⚙️ Configuración":
        show_configuration()

def show_executive_dashboard(df_empleados, df_obras, df_asistencias):
    st.markdown('<div class="section-header">📊 Dashboard Ejecutivo - Vista Power BI</div>', unsafe_allow_html=True)
    
    # Primera fila - Métricas estratégicas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        costo_total = df_empleados[df_empleados['activo']]['salario'].sum()
        st.metric("💰 Costo Nómina Mensual", f"${costo_total:,.0f}")
    
    with col2:
        horas_extra_totales = df_asistencias['horas_extra'].sum()
        st.metric("⏰ Horas Extra Acumuladas", f"{horas_extra_totales} h")
    
    with col3:
        ausentismo_promedio = df_empleados['ausencias_ultimo_mes'].mean()
        st.metric("🏥 Ausentismo Promedio", f"{ausentismo_promedio:.1f} días")
    
    with col4:
        evaluacion_promedio = df_empleados['evaluacion_desempeno'].mean()
        st.metric("⭐ Evaluación Desempeño", f"{evaluacion_promedio:.1f}%")
    
    # Segunda fila - Gráficos avanzados
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌐 Mapa de Calor - Productividad por Depto/Ubicación")
        
        # Crear matriz de productividad
        heatmap_data = df_asistencias.merge(df_empleados, left_on='empleado_id', right_on='id')
        pivot_table = heatmap_data.pivot_table(
            values='productividad', 
            index='departamento', 
            columns='ubicacion', 
            aggfunc='mean'
        ).fillna(0)
        
        fig = create_advanced_plotly_chart(
            pivot_table.reset_index(),
            'Productividad Promedio por Departamento y Ubicación',
            'density_heatmap',
            x='ubicacion',
            y='departamento',
            z=pivot_table.values.flatten(),
            color_continuous_scale='Viridis'
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Sunburst - Distribución Jerárquica")
        
        sunburst_data = df_empleados[df_empleados['activo']].copy()
        fig = create_advanced_plotly_chart(
            sunburst_data,
            'Distribución de Empleados por Departamento y Especialidad',
            'sunburst',
            path=['departamento', 'especialidad'],
            values='salario',
            color='salario',
            color_continuous_scale='Blues'
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Tercera fila - Más visualizaciones
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎻 Distribución Salarial - Violin Plot")
        
        fig = create_advanced_plotly_chart(
            df_empleados[df_empleados['activo']],
            'Distribución Salarial por Departamento',
            'violin',
            x='departamento',
            y='salario',
            color='departamento',
            box=True
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Tendencia Temporal - Productividad")
        
        df_asistencias['fecha'] = pd.to_datetime(df_asistencias['fecha'])
        df_asistencias['mes'] = df_asistencias['fecha'].dt.to_period('M').astype(str)
        
        productividad_mensual = df_asistencias.groupby('mes')['productividad'].mean().reset_index()
        
        fig = create_advanced_plotly_chart(
            productividad_mensual,
            'Evolución Mensual de Productividad',
            'line',
            x='mes',
            y='productividad',
            markers=True
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

def show_aptitude_analysis(df_empleados, df_obras):
    st.markdown('<div class="section-header">🎯 Análisis de Aptitud para Obras</div>', unsafe_allow_html=True)
    
    # Filtros para análisis de aptitud
    col1, col2, col3 = st.columns(3)
    
    with col1:
        obra_seleccionada = st.selectbox(
            "🏗️ Seleccionar Obra para Análisis",
            options=df_obras['nombre'].tolist(),
            index=0
        )
    
    with col2:
        departamento_filtro = st.selectbox(
            "🏢 Departamento",
            options=['Todos'] + df_empleados['departamento'].unique().tolist(),
            index=0
        )
    
    with col3:
        aptitud_filtro = st.selectbox(
            "✅ Estado Aptitud",
            options=['Todos', 'Aptos', 'No Aptos'],
            index=0
        )
    
    # Obtener datos de la obra seleccionada
    obra_info = df_obras[df_obras['nombre'] == obra_seleccionada].iloc[0]
    
    # Mostrar requisitos de la obra
    st.subheader(f"📋 Requisitos de la Obra: {obra_info['nombre']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info(f"**Tipo:** {obra_info['tipo']}")
        st.info(f"**Complejidad:** {obra_info['complejidad']}")
    
    with col2:
        st.info(f"**Apto Obra Compleja:** {'✅ Sí' if obra_info['requiere_apto_obra_compleja'] else '❌ No'}")
        st.info(f"**Exp. Mínima:** {obra_info['experiencia_minima_meses']} meses")
    
    with col3:
        st.info(f"**Vehículo Requerido:** {'✅ Sí' if obra_info['requiere_vehiculo'] else '❌ No'}")
        st.info(f"**Zona de Riesgo:** {'⚠️ Sí' if obra_info['zona_riesgo'] else '✅ No'}")
    
    with col4:
        st.info(f"**Ubicación:** {obra_info['ubicacion']}")
        st.info(f"**Presupuesto:** ${obra_info['presupuesto']:,.0f}")
    
    # Filtrar empleados según aptitud
    empleados_filtrados = df_empleados[df_empleados['activo']].copy()
    
    if departamento_filtro != 'Todos':
        empleados_filtrados = empleados_filtrados[empleados_filtrados['departamento'] == departamento_filtro]
    
    # Calcular aptitud para la obra seleccionada
    def calcular_aptitud(empleado, obra):
        criterios_cumplidos = 0
        criterios_totales = 4
        
        # Criterio 1: Aptitud para obra compleja
        if not obra['requiere_apto_obra_compleja'] or empleado['apto_obra_compleja']:
            criterios_cumplidos += 1
        
        # Criterio 2: Experiencia mínima
        if empleado['experiencia_meses'] >= obra['experiencia_minima_meses']:
            criterios_cumplidos += 1
        
        # Criterio 3: Vehículo propio (si se requiere)
        if not obra['requiere_vehiculo'] or empleado['vehiculo_propio']:
            criterios_cumplidos += 1
        
        # Criterio 4: Evaluación de desempeño
        if empleado['evaluacion_desempeno'] >= 70:
            criterios_cumplidos += 1
        
        porcentaje_aptitud = (criterios_cumplidos / criterios_totales) * 100
        return porcentaje_aptitud, criterios_cumplidos
    
    # Aplicar cálculo de aptitud
    aptitudes = []
    for _, emp in empleados_filtrados.iterrows():
        aptitud, criterios = calcular_aptitud(emp, obra_info)
        aptitudes.append({
            'empleado': emp,
            'porcentaje_aptitud': aptitud,
            'criterios_cumplidos': criterios,
            'apto': aptitud >= 75
        })
    
    # Filtrar por aptitud si se seleccionó
    if aptitud_filtro == 'Aptos':
        aptitudes = [apt for apt in aptitudes if apt['apto']]
    elif aptitud_filtro == 'No Aptos':
        aptitudes = [apt for apt in aptitudes if not apt['apto']]
    
    # Mostrar resultados
    st.subheader(f"👥 Empleados {aptitud_filtro} - {len(aptitudes)} encontrados")
    
    # Métricas de aptitud
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_aptos = sum(1 for apt in aptitudes if apt['apto'])
        st.metric("✅ Total Aptos", total_aptos)
    
    with col2:
        aptitud_promedio = np.mean([apt['porcentaje_aptitud'] for apt in aptitudes])
        st.metric("📊 Aptitud Promedio", f"{aptitud_promedio:.1f}%")
    
    with col3:
        criterios_promedio = np.mean([apt['criterios_cumplidos'] for apt in aptitudes])
        st.metric("🎯 Criterios Cumplidos", f"{criterios_promedio:.1f}/4")
    
    with col4:
        porcentaje_aptos = (total_aptos / len(aptitudes)) * 100 if aptitudes else 0
        st.metric("📈 % de Aptos", f"{porcentaje_aptos:.1f}%")
    
    # Mostrar empleados con tarjetas
    st.subheader("📋 Detalle de Empleados")
    
    for aptitud in aptitudes:
        emp = aptitud['empleado']
        card_class = "apto-card" if aptitud['apto'] else "no-apto-card"
        
        st.markdown(f'<div class="employee-card {card_class}">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            st.write(f"**{emp['nombre']} {emp['apellido']}**")
            st.write(f"*{emp['especialidad']} - {emp['departamento']}*")
            st.write(f"📅 Exp: {emp['experiencia_meses']} meses | 🎂 Edad: {emp['edad']} años")
        
        with col2:
            st.write(f"📊 Evaluación: {emp['evaluacion_desempeno']:.1f}%")
            st.write(f"🎓 Certificaciones: {emp['certificaciones']}")
            st.write(f"🚗 Vehículo: {'✅ Sí' if emp['vehiculo_propio'] else '❌ No'}")
        
        with col3:
            aptitud_color = "🟢" if aptitud['apto'] else "🔴"
            st.write(f"**{aptitud_color} Aptitud: {aptitud['porcentaje_aptitud']:.0f}%**")
            st.write(f"✅ {aptitud['criterios_cumplidos']}/4 criterios")
        
        with col4:
            if aptitud['apto']:
                st.success("**APTO**")
                if st.button("📋 Asignar", key=f"asignar_{emp['id']}"):
                    st.success(f"✅ {emp['nombre']} asignado a {obra_seleccionada}")
            else:
                st.error("**NO APTO**")
                st.button("📋 Asignar", key=f"asignar_{emp['id']}", disabled=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Gráfico de distribución de aptitud
    if aptitudes:
        st.subheader("📊 Análisis de Aptitud")
        
        aptitud_data = pd.DataFrame([{
            'Aptitud': apt['porcentaje_aptitud'],
            'Departamento': apt['empleado']['departamento'],
            'Apto': 'Apto' if apt['apto'] else 'No Apto'
        } for apt in aptitudes])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_advanced_plotly_chart(
                aptitud_data,
                'Distribución de Niveles de Aptitud',
                'histogram',
                x='Aptitud',
                color='Apto',
                nbins=20
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            dept_aptitud = aptitud_data.groupby('Departamento')['Aptitud'].mean().reset_index()
            fig = create_advanced_plotly_chart(
                dept_aptitud,
                'Aptitud Promedio por Departamento',
                'bar',
                x='Departamento',
                y='Aptitud',
                color='Aptitud',
                color_continuous_scale='RdYlGn'
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

def show_dashboard_manual():
    st.markdown('<div class="section-header">📖 Manual del Dashboard RRHH Analytics Pro</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="manual-section">
    <h3>🎯 Descripción General</h3>
    <p>El <strong>RRHH Analytics Pro</strong> es un sistema integral de gestión de recursos humanos diseñado para la industria de la construcción. 
    Combina análisis avanzados, visualizaciones interactivas y herramientas de gestión para optimizar la fuerza laboral.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Módulos del Dashboard
    st.subheader("📋 Módulos Disponibles")
    
    modules_info = {
        "📊 Dashboard Ejecutivo": {
            "descripción": "Vista general con métricas clave y visualizaciones ejecutivas",
            "insights": [
                "Tendencias de productividad en tiempo real",
                "Distribución de costos por departamento", 
                "Análisis comparativo entre ubicaciones",
                "Evolución temporal de indicadores clave"
            ],
            "visualizaciones": ["Mapas de calor", "Sunburst charts", "Violin plots", "Gráficos de tendencia"]
        },
        "👥 Gestión de Personal": {
            "descripción": "Gestión completa del capital humano con filtros avanzados",
            "insights": [
                "Composición de la fuerza laboral por departamento",
                "Análisis de compensación y equidad salarial",
                "Distribución de habilidades y certificaciones",
                "Segmentación por nivel educativo y experiencia"
            ],
            "visualizaciones": ["Tablas interactivas", "Gráficos de barras", "Scatter plots", "Box plots"]
        },
        "🏗️ Gestión de Obras": {
            "descripción": "Seguimiento y control de proyectos de construcción",
            "insights": [
                "Estado y progreso de obras activas",
                "Asignación óptima de recursos por proyecto",
                "Análisis de riesgos y alertas tempranas",
                "Control de presupuestos y cronogramas"
            ],
            "visualizaciones": ["Tarjetas de proyecto", "Gráficos de estado", "Métricas de progreso"]
        },
        "🎯 Aptitud para Obras": {
            "descripción": "Sistema inteligente de matching empleado-obra",
            "insights": [
                "Evaluación automática de compatibilidad",
                "Identificación de brechas de habilidades",
                "Optimización de asignaciones",
                "Análisis de criterios de aptitud"
            ],
            "visualizaciones": ["Tarjetas de aptitud", "Histogramas de distribución", "Gráficos comparativos"]
        },
        "📈 Analytics Avanzado": {
            "descripción": "Análisis predictivo y segmentación avanzada",
            "insights": [
                "Predicción de rotación voluntaria",
                "Segmentación por desempeño y potencial",
                "Análisis de correlaciones entre variables",
                "Identificación de patrones de comportamiento"
            ],
            "visualizaciones": ["Matrices de correlación", "Scatter plots", "Gráficos de dispersión"]
        },
        "⚠️ Sistema de Alertas": {
            "descripción": "Monitoreo proactivo de riesgos y oportunidades",
            "insights": [
                "Detección temprana de problemas de rendimiento",
                "Alertas de rotación en departamentos críticos",
                "Monitoreo de cumplimiento de metas",
                "Identificación de oportunidades de mejora"
            ],
            "visualizaciones": ["Alertas codificadas por color", "Paneles de control", "Indicadores de riesgo"]
        }
    }
    
    for module, info in modules_info.items():
        with st.expander(f"{module} - {info['descripción']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔍 Insights Principales:**")
                for insight in info['insights']:
                    st.write(f"• {insight}")
            
            with col2:
                st.write("**📊 Visualizaciones:**")
                for viz in info['visualizaciones']:
                    st.write(f"• {viz}")
    
    # Guías de Uso
    st.subheader("🛠️ Guías de Uso Rápido")
    
    usage_guides = {
        "Filtros Avanzados": "Utiliza los filtros multinivel para segmentar datos específicos por departamento, ubicación, aptitud, etc.",
        "Visualizaciones Interactivas": "Haz hover sobre los gráficos para ver detalles específicos. Usa zoom en gráficos complejos.",
        "Exportación de Datos": "Todos los dataframes son exportables haciendo clic en el ícono de exportación.",
        "Alertas Inteligentes": "Configura umbrales personalizados para recibir alertas proactivas.",
        "Sistema de Aptitud": "Selecciona una obra específica para analizar la compatibilidad automática con empleados."
    }
    
    for guide, description in usage_guides.items():
        st.info(f"**{guide}:** {description}")
    
    # KPIs y Métricas Explicadas
    st.subheader("📈 Explicación de Métricas Clave")
    
    kpis_explained = {
        "Productividad": "Mide la eficiencia del trabajo realizado vs. tiempo invertido. Meta: >85%",
        "Rotación": "Porcentaje de empleados que dejan la empresa. Meta: <8%", 
        "Aptitud Obra Compleja": "Porcentaje de empleados calificados para obras de alta complejidad",
        "Costo por Hora": "Costo laboral promedio por hora trabajada",
        "Ausentismo": "Días de ausencia no programados por empleado/mes. Meta: <3 días",
        "Evaluación Desempeño": "Calificación promedio en evaluaciones de desempeño. Meta: >80%"
    }
    
    for kpi, explanation in kpis_explained.items():
        st.write(f"**{kpi}:** {explanation}")
    
    # Consejos para Análisis
    st.subheader("💡 Consejos para Análisis Efectivo")
    
    tips = [
        "**Compara departamentos** para identificar mejores prácticas y oportunidades de mejora",
        "**Monitorea tendencias temporales** para detectar patrones estacionales o cambios graduales",
        "**Combina múltiples métricas** para obtener una visión holística del desempeño",
        "**Utiliza el sistema de aptitud** para optimizar asignaciones y reducir riesgos",
        "**Configura alertas personalizadas** para monitoreo proactivo de indicadores críticos",
        "**Exporta datos específicos** para análisis más profundos en otras herramientas"
    ]
    
    for tip in tips:
        st.write(f"• {tip}")

# Funciones placeholder para los módulos faltantes
def show_person_management(df_empleados, df_asistencias):
    st.markdown('<div class="section-header">👥 Gestión de Personal</div>', unsafe_allow_html=True)
    st.info("Módulo de Gestión de Personal - En desarrollo")
    
def show_project_management(df_obras, df_asistencias, df_empleados):
    st.markdown('<div class="section-header">🏗️ Gestión de Obras</div>', unsafe_allow_html=True)
    st.info("Módulo de Gestión de Obras - En desarrollo")
    
def show_advanced_analytics(df_empleados, df_asistencias):
    st.markdown('<div class="section-header">📈 Analytics Avanzado</div>', unsafe_allow_html=True)
    st.info("Módulo de Analytics Avanzado - En desarrollo")
    
def show_early_warnings(df_empleados, df_obras, df_asistencias):
    st.markdown('<div class="section-header">⚠️ Sistema de Alertas</div>', unsafe_allow_html=True)
    st.info("Módulo de Sistema de Alertas - En desarrollo")
    
def show_configuration():
    st.markdown('<div class="section-header">⚙️ Configuración</div>', unsafe_allow_html=True)
    st.info("Módulo de Configuración - En desarrollo")

if __name__ == "__main__":
    main()
