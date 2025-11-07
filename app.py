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
    .project-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3498db;
    }
    .risk-high {
        border-left: 4px solid #e74c3c !important;
    }
    .risk-medium {
        border-left: 4px solid #f39c12 !important;
    }
    .risk-low {
        border-left: 4px solid #2ecc71 !important;
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

def show_person_management(df_empleados, df_asistencias):
    st.markdown('<div class="section-header">👥 Gestión de Personal</div>', unsafe_allow_html=True)
    
    # Filtros
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dept_filter = st.selectbox(
            "🏢 Departamento",
            options=['Todos'] + df_empleados['departamento'].unique().tolist()
        )
    
    with col2:
        ubicacion_filter = st.selectbox(
            "📍 Ubicación",
            options=['Todos'] + df_empleados['ubicacion'].unique().tolist()
        )
    
    with col3:
        estado_filter = st.selectbox(
            "✅ Estado",
            options=['Todos', 'Activos', 'Inactivos']
        )
    
    with col4:
        aptitud_filter = st.selectbox(
            "🎯 Aptitud Obra Compleja",
            options=['Todos', 'Aptos', 'No Aptos']
        )
    
    # Aplicar filtros
    filtered_employees = df_empleados.copy()
    
    if dept_filter != 'Todos':
        filtered_employees = filtered_employees[filtered_employees['departamento'] == dept_filter]
    
    if ubicacion_filter != 'Todos':
        filtered_employees = filtered_employees[filtered_employees['ubicacion'] == ubicacion_filter]
    
    if estado_filter == 'Activos':
        filtered_employees = filtered_employees[filtered_employees['activo'] == True]
    elif estado_filter == 'Inactivos':
        filtered_employees = filtered_employees[filtered_employees['activo'] == False]
    
    if aptitud_filter == 'Aptos':
        filtered_employees = filtered_employees[filtered_employees['apto_obra_compleja'] == True]
    elif aptitud_filter == 'No Aptos':
        filtered_employees = filtered_employees[filtered_employees['apto_obra_compleja'] == False]
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Filtrado", len(filtered_employees))
    
    with col2:
        avg_salary = filtered_employees['salario'].mean()
        st.metric("💰 Salario Promedio", f"${avg_salary:,.0f}")
    
    with col3:
        avg_experience = filtered_employees['experiencia_meses'].mean()
        st.metric("📅 Experiencia Promedio", f"{avg_experience:.0f} meses")
    
    with col4:
        avg_performance = filtered_employees['evaluacion_desempeno'].mean()
        st.metric("⭐ Desempeño Promedio", f"{avg_performance:.1f}%")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución por departamento
        dept_dist = filtered_employees['departamento'].value_counts()
        fig = px.pie(
            values=dept_dist.values,
            names=dept_dist.index,
            title='Distribución por Departamento'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Salario vs Experiencia
        fig = px.scatter(
            filtered_employees,
            x='experiencia_meses',
            y='salario',
            color='departamento',
            title='Salario vs Experiencia por Departamento',
            size='evaluacion_desempeno',
            hover_data=['nombre', 'apellido']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de empleados
    st.subheader("📋 Lista de Empleados")
    
    # Seleccionar columnas para mostrar
    display_columns = ['id', 'nombre', 'apellido', 'departamento', 'cargo', 'salario', 
                      'experiencia_meses', 'evaluacion_desempeno', 'apto_obra_compleja']
    
    st.dataframe(
        filtered_employees[display_columns],
        use_container_width=True,
        height=400
    )

def show_project_management(df_obras, df_asistencias, df_empleados):
    st.markdown('<div class="section-header">🏗️ Gestión de Obras</div>', unsafe_allow_html=True)
    
    # Filtros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        estado_filter = st.selectbox(
            "📊 Estado Obra",
            options=['Todos'] + df_obras['estado'].unique().tolist(),
            key="estado_obra"
        )
    
    with col2:
        tipo_filter = st.selectbox(
            "🏢 Tipo Obra",
            options=['Todos'] + df_obras['tipo'].unique().tolist()
        )
    
    with col3:
        complejidad_filter = st.selectbox(
            "⚡ Complejidad",
            options=['Todos'] + df_obras['complejidad'].unique().tolist()
        )
    
    # Aplicar filtros
    filtered_projects = df_obras.copy()
    
    if estado_filter != 'Todos':
        filtered_projects = filtered_projects[filtered_projects['estado'] == estado_filter]
    
    if tipo_filter != 'Todos':
        filtered_projects = filtered_projects[filtered_projects['tipo'] == tipo_filter]
    
    if complejidad_filter != 'Todos':
        filtered_projects = filtered_projects[filtered_projects['complejidad'] == complejidad_filter]
    
    # Métricas de obras
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_presupuesto = filtered_projects['presupuesto'].sum()
        st.metric("💰 Presupuesto Total", f"${total_presupuesto:,.0f}")
    
    with col2:
        obras_en_progreso = len(filtered_projects[filtered_projects['estado'] == 'En Progreso'])
        st.metric("🏗️ Obras en Progreso", obras_en_progreso)
    
    with col3:
        obras_en_riesgo = len(filtered_projects[filtered_projects['estado'] == 'En Riesgo'])
        st.metric("⚠️ Obras en Riesgo", obras_en_riesgo)
    
    with col4:
        avg_duration = filtered_projects['duracion_estimada'].mean()
        st.metric("📅 Duración Promedio", f"{avg_duration:.0f} días")
    
    # Mostrar obras como tarjetas
    st.subheader("📋 Detalle de Obras")
    
    for _, obra in filtered_projects.iterrows():
        # Determinar clase de riesgo
        if obra['estado'] == 'En Riesgo':
            risk_class = "risk-high"
        elif obra['estado'] == 'En Progreso':
            risk_class = "risk-medium"
        else:
            risk_class = "risk-low"
        
        st.markdown(f'<div class="project-card {risk_class}">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            st.write(f"### {obra['nombre']}")
            st.write(f"**Ubicación:** {obra['ubicacion']} | **Gerente:** {obra['gerente']}")
            st.write(f"**Tipo:** {obra['tipo']} | **Complejidad:** {obra['complejidad']}")
        
        with col2:
            st.write(f"**Presupuesto:** ${obra['presupuesto']:,.0f}")
            st.write(f"**Duración:** {obra['duracion_estimada']} días")
            st.write(f"**Inicio:** {obra['fecha_inicio'].strftime('%d/%m/%Y')}")
        
        with col3:
            st.write(f"**Estado:** {obra['estado']}")
            st.write(f"**Apto Compleja:** {'✅' if obra['requiere_apto_obra_compleja'] else '❌'}")
            st.write(f"**Exp. Mínima:** {obra['experiencia_minima_meses']} meses")
        
        with col4:
            status_color = {
                'En Planificación': '🟡',
                'En Progreso': '🟢',
                'En Riesgo': '🔴',
                'Completado': '🔵',
                'Pausado': '🟠'
            }[obra['estado']]
            st.write(f"### {status_color}")
            
            if st.button("📊 Detalles", key=f"detalles_{obra['id']}"):
                st.session_state[f"show_details_{obra['id']}"] = True
        
        # Mostrar detalles si se hace clic
        if st.session_state.get(f"show_details_{obra['id']}", False):
            st.info(f"Detalles completos de {obra['nombre']}")
            # Aquí podrías mostrar más información específica de la obra
    
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Gráficos de análisis de obras
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de presupuesto por tipo
        fig = px.bar(
            filtered_projects.groupby('tipo')['presupuesto'].sum().reset_index(),
            x='tipo',
            y='presupuesto',
            title='Presupuesto por Tipo de Obra',
            color='tipo'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Estado de obras
        estado_counts = filtered_projects['estado'].value_counts()
        fig = px.pie(
            values=estado_counts.values,
            names=estado_counts.index,
            title='Distribución de Estados de Obras'
        )
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

def show_advanced_analytics(df_empleados, df_asistencias):
    st.markdown('<div class="section-header">📈 Analytics Avanzado</div>', unsafe_allow_html=True)
    
    # Análisis predictivo de rotación
    st.subheader("🔮 Predicción de Rotación")
    
    # Simular análisis predictivo
    df_analytics = df_empleados[df_empleados['activo']].copy()
    
    # Crear características para el modelo (simulado)
    df_analytics['riesgo_rotacion'] = np.random.normal(0.3, 0.2, len(df_analytics))
    df_analytics['riesgo_rotacion'] = df_analytics['riesgo_rotacion'].clip(0, 1)
    
    # Clasificar riesgo
    def clasificar_riesgo(score):
        if score > 0.7:
            return 'Alto'
        elif score > 0.4:
            return 'Medio'
        else:
            return 'Bajo'
    
    df_analytics['nivel_riesgo'] = df_analytics['riesgo_rotacion'].apply(clasificar_riesgo)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alto_riesgo = len(df_analytics[df_analytics['nivel_riesgo'] == 'Alto'])
        st.metric("🔴 Alto Riesgo", alto_riesgo)
    
    with col2:
        medio_riesgo = len(df_analytics[df_analytics['nivel_riesgo'] == 'Medio'])
        st.metric("🟡 Medio Riesgo", medio_riesgo)
    
    with col3:
        bajo_riesgo = len(df_analytics[df_analytics['nivel_riesgo'] == 'Bajo'])
        st.metric("🟢 Bajo Riesgo", bajo_riesgo)
    
    # Gráficos de análisis avanzado
    col1, col2 = st.columns(2)
    
    with col1:
        # Matriz de correlación
        numeric_cols = ['edad', 'experiencia_meses', 'salario', 'evaluacion_desempeno', 'ausencias_ultimo_mes']
        corr_matrix = df_analytics[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title='Matriz de Correlación entre Variables',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Segmentación por desempeño y potencial
        fig = px.scatter(
            df_analytics,
            x='evaluacion_desempeno',
            y='experiencia_meses',
            color='nivel_riesgo',
            size='salario',
            title='Segmentación: Desempeño vs Experiencia',
            hover_data=['nombre', 'apellido', 'departamento'],
            color_discrete_map={'Alto': 'red', 'Medio': 'orange', 'Bajo': 'green'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de clusters
    st.subheader("🎯 Segmentación Avanzada")
    
    # Simular clusters
    df_analytics['cluster'] = np.random.choice(['A - Alto Potencial', 'B - Estables', 'C - Necesitan Soporte'], 
                                              len(df_analytics), p=[0.2, 0.6, 0.2])
    
    col1, col2 = st.columns(2)
    
    with col1:
        cluster_counts = df_analytics['cluster'].value_counts()
        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            title='Distribución de Segmentos',
            color=cluster_counts.index,
            labels={'x': 'Segmento', 'y': 'Cantidad'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Análisis de composición salarial por segmento
        fig = px.box(
            df_analytics,
            x='cluster',
            y='salario',
            title='Distribución Salarial por Segmento',
            color='cluster'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_early_warnings(df_empleados, df_obras, df_asistencias):
    st.markdown('<div class="section-header">⚠️ Sistema de Alertas Tempranas</div>', unsafe_allow_html=True)
    
    # Alertas de empleados
    st.subheader("👥 Alertas de Personal")
    
    # Generar alertas simuladas
    alertas_empleados = []
    
    # Alertas por bajo desempeño
    bajo_desempeno = df_empleados[
        (df_empleados['activo']) & 
        (df_empleados['evaluacion_desempeno'] < 70)
    ]
    for _, emp in bajo_desempeno.iterrows():
        alertas_empleados.append({
            'tipo': 'Bajo Desempeño',
            'nivel': 'Alto',
            'descripcion': f"{emp['nombre']} {emp['apellido']} - Evaluación: {emp['evaluacion_desempeno']:.1f}%",
            'departamento': emp['departamento']
        })
    
    # Alertas por alto ausentismo
    alto_ausentismo = df_empleados[
        (df_empleados['activo']) & 
        (df_empleados['ausencias_ultimo_mes'] > 3)
    ]
    for _, emp in alto_ausentismo.iterrows():
        alertas_empleados.append({
            'tipo': 'Alto Ausentismo',
            'nivel': 'Medio',
            'descripcion': f"{emp['nombre']} {emp['apellido']} - {emp['ausencias_ultimo_mes']} ausencias/mes",
            'departamento': emp['departamento']
        })
    
    # Mostrar alertas de empleados
    for alerta in alertas_empleados:
        if alerta['nivel'] == 'Alto':
            st.markdown(f'<div class="alert-high">', unsafe_allow_html=True)
        elif alerta['nivel'] == 'Medio':
            st.markdown(f'<div class="alert-medium">', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-low">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.write(f"**{alerta['tipo']}**")
            st.write(f"Departamento: {alerta['departamento']}")
        
        with col2:
            st.write(alerta['descripcion'])
        
        with col3:
            if st.button("📋 Acción", key=f"accion_{alerta['descripcion']}"):
                st.success(f"Acción tomada para {alerta['descripcion']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Alertas de obras
    st.subheader("🏗️ Alertas de Obras")
    
    alertas_obras = []
    
    # Obras en riesgo
    obras_riesgo = df_obras[df_obras['estado'] == 'En Riesgo']
    for _, obra in obras_riesgo.iterrows():
        alertas_obras.append({
            'tipo': 'Obra en Riesgo',
            'nivel': 'Alto',
            'descripcion': f"{obra['nombre']} - {obra['ubicacion']}",
            'presupuesto': obra['presupuesto']
        })
    
    # Obras sin gerente asignado (simulado)
    for _, obra in df_obras.sample(2).iterrows():
        alertas_obras.append({
            'tipo': 'Falta Recursos',
            'nivel': 'Medio',
            'descripcion': f"{obra['nombre']} - Necesita más personal especializado",
            'presupuesto': obra['presupuesto']
        })
    
    # Mostrar alertas de obras
    for alerta in alertas_obras:
        if alerta['nivel'] == 'Alto':
            st.markdown(f'<div class="alert-high">', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-medium">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.write(f"**{alerta['tipo']}**")
            st.write(f"Presupuesto: ${alerta['presupuesto']:,.0f}")
        
        with col2:
            st.write(alerta['descripcion'])
        
        with col3:
            if st.button("🔧 Resolver", key=f"resolver_{alerta['descripcion']}"):
                st.success(f"Problema resuelto para {alerta['descripcion']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Métricas de alertas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Alertas", len(alertas_empleados) + len(alertas_obras))
    
    with col2:
        alertas_altas = len([a for a in alertas_empleados + alertas_obras if a['nivel'] == 'Alto'])
        st.metric("🔴 Alertas Altas", alertas_altas)
    
    with col3:
        alertas_medias = len([a for a in alertas_empleados + alertas_obras if a['nivel'] == 'Medio'])
        st.metric("🟡 Alertas Medias", alertas_medias)
    
    with col4:
        st.metric("✅ Resueltas Hoy", np.random.randint(2, 8))

def show_configuration():
    st.markdown('<div class="section-header">⚙️ Configuración del Sistema</div>', unsafe_allow_html=True)
    
    # Configuración de parámetros
    st.subheader("📋 Parámetros del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("Umbral Bajo Desempeño (%)", min_value=0, max_value=100, value=70)
        st.number_input("Umbral Alto Ausentismo (días/mes)", min_value=1, max_value=30, value=3)
        st.number_input("Porcentaje Mínimo Aptitud", min_value=0, max_value=100, value=75)
    
    with col2:
        st.number_input("Horas Extra Máximas Semanales", min_value=1, max_value=20, value=10)
        st.number_input("Experiencia Mínima Obra Compleja (meses)", min_value=1, max_value=60, value=24)
        st.number_input("Evaluación Mínima Promoción", min_value=0, max_value=100, value=80)
    
    # Configuración de notificaciones
    st.subheader("🔔 Configuración de Notificaciones")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.checkbox("Alertas de Bajo Desempeño", value=True)
        st.checkbox("Alertas de Alto Ausentismo", value=True)
        st.checkbox("Alertas de Rotación", value=True)
    
    with col2:
        st.checkbox("Notificaciones de Obras en Riesgo", value=True)
        st.checkbox("Reportes Semanales Automáticos", value=True)
        st.checkbox("Recordatorios de Evaluaciones", value=True)
    
    with col3:
        st.selectbox("Frecuencia de Reportes", ["Diario", "Semanal", "Mensual"])
        st.selectbox("Método de Notificación", ["Email", "SMS", "Ambos"])
        st.text_input("Email de Contacto", "admin@empresa.com")
    
    # Configuración de integraciones
    st.subheader("🔗 Integraciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("API Key Sistema de Nómina")
        st.text_input("URL Base de Datos")
        st.text_input("Token de Autenticación")
    
    with col2:
        st.checkbox("Sincronización Automática", value=True)
        st.number_input("Intervalo Sincronización (min)", min_value=5, max_value=1440, value=60)
        st.selectbox("Nivel de Log", ["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # Acciones del sistema
    st.subheader("🛠️ Acciones del Sistema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Sincronizar Datos", use_container_width=True):
            st.success("Datos sincronizados correctamente")
        
        if st.button("📊 Generar Reporte", use_container_width=True):
            st.success("Reporte generado y enviado")
    
    with col2:
        if st.button("💾 Respaldar Base", use_container_width=True):
            st.success("Respaldo completado exitosamente")
        
        if st.button("🧹 Limpiar Cache", use_container_width=True):
            st.success("Cache limpiado correctamente")
    
    with col3:
        if st.button("🔍 Ver Logs", use_container_width=True):
            st.info("Mostrando logs del sistema...")
        
        if st.button("🔄 Reiniciar Sistema", use_container_width=True):
            st.warning("Reiniciando sistema...")

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

if __name__ == "__main__":
    main()
