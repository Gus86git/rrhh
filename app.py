import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importaciones condicionales para gráficos
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib/Seaborn no disponibles. Algunos gráficos no se mostrarán.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly no disponible. Gráficos interactivos no se mostrarán.")

# Configuración de página
st.set_page_config(
    page_title="RRHH Analytics Pro",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .alert-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background: linear-gradient(135deg, #ffd93d 0%, #ffcd3c 100%);
        color: black;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Generar datos sintéticos completos para la demo"""
    np.random.seed(42)
    
    # Generar empleados
    nombres = ['Sofia', 'Martina', 'Lucia', 'Ana', 'Carolina', 'Valentina', 
               'Carlos', 'Diego', 'Juan', 'Pablo', 'Ricardo', 'Javier', 
               'Miguel', 'Roberto', 'Fernando']
    apellidos = ['Lopez', 'Gonzalez', 'Garcia', 'Martinez', 'Rodriguez', 
                 'Perez', 'Diaz', 'Gomez', 'Fernandez', 'Romero']
    
    empleados = []
    for i in range(150):
        genero = np.random.choice(['Femenino', 'Masculino'], p=[0.35, 0.65])
        dept = np.random.choice(['Albañilería', 'Electricidad', 'Plomería', 'Herrería', 'Pintura'], 
                               p=[0.3, 0.25, 0.2, 0.15, 0.1])
        
        # Salarios base por departamento
        salario_base = {
            'Albañilería': 80000, 'Electricidad': 95000, 
            'Plomería': 85000, 'Herrería': 110000, 'Pintura': 75000
        }[dept]
        
        salario = salario_base * np.random.uniform(0.8, 1.5)
        experiencia = np.random.randint(6, 120)
        
        empleados.append({
            'id': f"EMP{i+1:03d}",
            'nombre': np.random.choice(nombres),
            'apellido': np.random.choice(apellidos),
            'genero': genero,
            'edad': np.random.randint(22, 60),
            'departamento': dept,
            'cargo': f"{dept} {'Senior' if experiencia > 60 else 'Junior'}",
            'salario': round(salario, 2),
            'fecha_contratacion': datetime.now() - timedelta(days=np.random.randint(30, 365*3)),
            'experiencia_meses': experiencia,
            'ubicacion': np.random.choice(['Sede Central', 'Obra Norte', 'Obra Sur', 'Obra Este']),
            'nivel_educacion': np.random.choice(['Secundario', 'Terciario', 'Universitario', 'Maestría'], 
                                              p=[0.4, 0.3, 0.2, 0.1]),
            'activo': np.random.choice([True, False], p=[0.92, 0.08])
        })
    
    df_empleados = pd.DataFrame(empleados)
    
    # Generar obras
    obras = []
    for i in range(12):
        obras.append({
            'id': f"OBR{i+1:03d}",
            'nombre': f"Proyecto {['Residencial', 'Comercial', 'Industrial'][i % 3]} {i+1}",
            'ubicacion': np.random.choice(['Nordelta', 'Pilar', 'Tigre', 'Escobar', 'San Isidro']),
            'presupuesto': np.random.randint(5000000, 25000000),
            'fecha_inicio': datetime.now() - timedelta(days=np.random.randint(30, 300)),
            'duracion_estimada': np.random.randint(90, 360),
            'estado': np.random.choice(['En Planificación', 'En Progreso', 'En Riesgo', 'Completado'], 
                                     p=[0.1, 0.6, 0.2, 0.1]),
            'gerente': np.random.choice([f"{emp['nombre']} {emp['apellido']}" for emp in empleados[:20]]),
            'complejidad': np.random.choice(['Baja', 'Media', 'Alta'], p=[0.3, 0.5, 0.2])
        })
    
    df_obras = pd.DataFrame(obras)
    
    # Generar asistencias y rendimiento
    asistencias = []
    for _ in range(2000):
        emp = empleados[np.random.randint(0, len(empleados))]
        obra = obras[np.random.randint(0, len(obras))]
        
        fecha = datetime.now() - timedelta(days=np.random.randint(1, 90))
        
        asistencias.append({
            'empleado_id': emp['id'],
            'obra_id': obra['id'],
            'fecha': fecha,
            'horas_trabajadas': np.random.randint(6, 10),
            'horas_extra': np.random.choice([0, 0, 0, 1, 2]),
            'productividad': np.random.normal(85, 10),
            'calidad_trabajo': np.random.normal(90, 5),
            'incidentes_seguridad': np.random.poisson(0.1),
            'ausente': np.random.choice([True, False], p=[0.05, 0.95])
        })
    
    df_asistencias = pd.DataFrame(asistencias)
    df_asistencias['productividad'] = df_asistencias['productividad'].clip(50, 100)
    df_asistencias['calidad_trabajo'] = df_asistencias['calidad_trabajo'].clip(70, 100)
    
    return df_empleados, df_obras, df_asistencias

def create_simple_plotly_chart(data, title, chart_type='bar', x=None, y=None, color=None):
    """Función helper para crear gráficos Plotly de manera segura"""
    if not PLOTLY_AVAILABLE:
        st.warning(f"Plotly no disponible para: {title}")
        return None
    
    try:
        if chart_type == 'bar':
            fig = px.bar(data, x=x, y=y, color=color, title=title)
        elif chart_type == 'pie':
            fig = px.pie(data, names=x, values=y, title=title)
        elif chart_type == 'scatter':
            fig = px.scatter(data, x=x, y=y, color=color, title=title)
        elif chart_type == 'line':
            fig = px.line(data, x=x, y=y, title=title)
        else:
            fig = px.bar(data, x=x, y=y, title=title)
        
        fig.update_layout(height=300)
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
         "📈 Analytics", "🎯 Desempeño", "⚠️ Alertas", "⚙️ Configuración"]
    )
    
    # KPIs Principales - Siempre visibles
    st.markdown("### 📈 Métricas Clave en Tiempo Real")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_empleados = len(df_empleados[df_empleados['activo']])
        st.metric("👥 Empleados Activos", total_empleados, delta="+5%")
    
    with col2:
        costo_total = df_empleados[df_empleados['activo']]['salario'].sum()
        st.metric("💰 Costo Mensual", f"${costo_total:,.0f}")
    
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
    elif menu == "📈 Analytics":
        show_advanced_analytics(df_empleados, df_asistencias)
    elif menu == "🎯 Desempeño":
        show_performance_analytics(df_empleados, df_asistencias)
    elif menu == "⚠️ Alertas":
        show_early_warnings(df_empleados, df_obras, df_asistencias)
    elif menu == "⚙️ Configuración":
        show_configuration()

def show_executive_dashboard(df_empleados, df_obras, df_asistencias):
    st.markdown('<div class="section-header">📊 Dashboard Ejecutivo</div>', unsafe_allow_html=True)
    
    # Primera fila de gráficos
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Distribución por Departamento")
        
        if PLOTLY_AVAILABLE:
            # Gráfico de distribución de empleados por departamento
            dept_count = df_empleados['departamento'].value_counts().reset_index()
            dept_count.columns = ['Departamento', 'Cantidad']
            
            fig = create_simple_plotly_chart(
                dept_count, 
                'Empleados por Departamento', 
                'bar', 
                x='Departamento', 
                y='Cantidad'
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback: mostrar tabla
            dept_count = df_empleados['departamento'].value_counts()
            st.dataframe(dept_count.reset_index().rename(columns={'index': 'Departamento', 'departamento': 'Empleados'}))
        
        # Métricas rápidas
        st.subheader("📋 Resumen Ejecutivo")
        col_met1, col_met2, col_met3 = st.columns(3)
        
        with col_met1:
            avg_salary = df_empleados['salario'].mean()
            st.metric("💰 Salario Promedio", f"${avg_salary:,.0f}")
        
        with col_met2:
            avg_age = df_empleados['edad'].mean()
            st.metric("🎂 Edad Promedio", f"{avg_age:.1f} años")
        
        with col_met3:
            avg_exp = df_empleados['experiencia_meses'].mean() / 12
            st.metric("📅 Exp. Promedio", f"{avg_exp:.1f} años")
    
    with col2:
        st.subheader("🎯 Distribución por Género")
        
        if PLOTLY_AVAILABLE:
            genero_count = df_empleados['genero'].value_counts().reset_index()
            genero_count.columns = ['Género', 'Cantidad']
            
            fig = create_simple_plotly_chart(
                genero_count,
                'Distribución por Género',
                'pie',
                x='Género',
                y='Cantidad'
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            genero_count = df_empleados['genero'].value_counts()
            for genero, count in genero_count.items():
                st.write(f"**{genero}:** {count} empleados")
                st.progress(count / len(df_empleados))
        
        # Estado de obras
        st.subheader("🏗️ Estado de Obras")
        obra_estado = df_obras['estado'].value_counts()
        for estado, count in obra_estado.items():
            st.write(f"**{estado}:** {count}")
            st.progress(count / len(df_obras))

def show_person_management(df_empleados, df_asistencias):
    st.markdown('<div class="section-header">👥 Gestión de Personal</div>', unsafe_allow_html=True)
    
    # Filtros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dept_filter = st.multiselect(
            "🏢 Departamento",
            options=df_empleados['departamento'].unique(),
            default=df_empleados['departamento'].unique()
        )
    
    with col2:
        ubicacion_filter = st.multiselect(
            "📍 Ubicación",
            options=df_empleados['ubicacion'].unique(),
            default=df_empleados['ubicacion'].unique()
        )
    
    with col3:
        estado_filter = st.selectbox(
            "📊 Estado",
            options=['Todos', 'Activos', 'Inactivos'],
            index=0
        )
    
    # Aplicar filtros
    empleados_filtrados = df_empleados.copy()
    
    if dept_filter:
        empleados_filtrados = empleados_filtrados[empleados_filtrados['departamento'].isin(dept_filter)]
    
    if ubicacion_filter:
        empleados_filtrados = empleados_filtrados[empleados_filtrados['ubicacion'].isin(ubicacion_filter)]
    
    if estado_filter == 'Activos':
        empleados_filtrados = empleados_filtrados[empleados_filtrados['activo'] == True]
    elif estado_filter == 'Inactivos':
        empleados_filtrados = empleados_filtrados[empleados_filtrados['activo'] == False]
    
    # Mostrar resultados
    st.metric("👥 Empleados Filtrados", len(empleados_filtrados))
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📋 Lista de Empleados")
        st.dataframe(
            empleados_filtrados[[
                'id', 'nombre', 'apellido', 'departamento', 'cargo', 
                'salario', 'experiencia_meses', 'ubicacion', 'activo'
            ]].rename(columns={
                'id': 'ID', 'nombre': 'Nombre', 'apellido': 'Apellido',
                'departamento': 'Departamento', 'cargo': 'Cargo',
                'salario': 'Salario', 'experiencia_meses': 'Exp (meses)',
                'ubicacion': 'Ubicación', 'activo': 'Activo'
            }),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.subheader("📊 Análisis del Personal")
        
        # Gráfico de salarios por departamento
        if PLOTLY_AVAILABLE:
            fig = create_simple_plotly_chart(
                empleados_filtrados,
                'Salarios por Departamento',
                'box',
                x='departamento',
                y='salario'
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Distribución por nivel educativo
        if PLOTLY_AVAILABLE:
            educ_count = empleados_filtrados['nivel_educacion'].value_counts().reset_index()
            educ_count.columns = ['Nivel Educativo', 'Cantidad']
            
            fig = create_simple_plotly_chart(
                educ_count,
                'Distribución por Nivel Educativo',
                'bar',
                x='Nivel Educativo',
                y='Cantidad'
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

def show_project_management(df_obras, df_asistencias, df_empleados):
    st.markdown('<div class="section-header">🏗️ Gestión de Obras</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Estado de Obras")
        
        for _, obra in df_obras.iterrows():
            with st.expander(f"🏗️ {obra['nombre']} - {obra['ubicacion']}"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write(f"**📊 Estado:** {obra['estado']}")
                    st.write(f"**🎯 Complejidad:** {obra['complejidad']}")
                    st.write(f"**👨‍💼 Gerente:** {obra['gerente']}")
                    st.write(f"**💰 Presupuesto:** ${obra['presupuesto']:,.0f}")
                
                with col_b:
                    # Calcular progreso
                    fecha_inicio = obra['fecha_inicio']
                    duracion = obra['duracion_estimada']
                    dias_transcurridos = (datetime.now() - fecha_inicio).days
                    progreso = min(95, max(5, (dias_transcurridos / duracion) * 100))
                    
                    st.write(f"**📈 Progreso:** {progreso:.1f}%")
                    st.progress(progreso/100)
                    
                    # Empleados en esta obra
                    empleados_obra = df_asistencias[df_asistencias['obra_id'] == obra['id']]['empleado_id'].nunique()
                    st.write(f"**👥 Empleados:** {empleados_obra}")
    
    with col2:
        st.subheader("📊 Métricas de Obras")
        
        # KPIs
        obras_en_progreso = len(df_obras[df_obras['estado'] == 'En Progreso'])
        obras_en_riesgo = len(df_obras[df_obras['estado'] == 'En Riesgo'])
        presupuesto_total = df_obras['presupuesto'].sum()
        
        st.metric("🔄 Obras en Progreso", obras_en_progreso)
        st.metric("⚠️ Obras en Riesgo", obras_en_riesgo)
        st.metric("💰 Presupuesto Total", f"${presupuesto_total:,.0f}")
        
        # Gráfico de estado de obras
        if PLOTLY_AVAILABLE:
            estado_count = df_obras['estado'].value_counts().reset_index()
            estado_count.columns = ['Estado', 'Cantidad']
            
            fig = create_simple_plotly_chart(
                estado_count,
                'Distribución por Estado',
                'pie',
                x='Estado',
                y='Cantidad'
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

def show_advanced_analytics(df_empleados, df_asistencias):
    st.markdown('<div class="section-header">📈 Analytics Avanzado</div>', unsafe_allow_html=True)
    
    # Combinar datos para análisis
    analytics_data = df_empleados.merge(
        df_asistencias.groupby('empleado_id').agg({
            'productividad': 'mean',
            'horas_extra': 'sum',
            'ausente': 'sum'
        }).reset_index(),
        left_on='id', right_on='empleado_id'
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Salario vs Experiencia")
        
        if PLOTLY_AVAILABLE:
            fig = create_simple_plotly_chart(
                analytics_data,
                'Salario vs Experiencia',
                'scatter',
                x='experiencia_meses',
                y='salario',
                color='departamento'
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Productividad por Departamento")
        
        if PLOTLY_AVAILABLE:
            dept_productividad = analytics_data.groupby('departamento')['productividad'].mean().reset_index()
            
            fig = create_simple_plotly_chart(
                dept_productividad,
                'Productividad Promedio por Departamento',
                'bar',
                x='departamento',
                y='productividad'
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # Análisis predictivo simulado
    st.subheader("🔮 Predicción de Riesgo de Rotación")
    
    # Simular predicciones
    empleados_riesgo = df_empleados.sample(5)
    
    for _, emp in empleados_riesgo.iterrows():
        riesgo = np.random.uniform(0.3, 0.9)
        
        if riesgo > 0.7:
            st.error(f"🚨 **{emp['nombre']} {emp['apellido']}** - Riesgo de rotación: {riesgo:.0%}")
        elif riesgo > 0.5:
            st.warning(f"⚠️ **{emp['nombre']} {emp['apellido']}** - Riesgo de rotación: {riesgo:.0%}")
        else:
            st.success(f"✅ **{emp['nombre']} {emp['apellido']}** - Riesgo de rotación: {riesgo:.0%}")

def show_performance_analytics(df_empleados, df_asistencias):
    st.markdown('<div class="section-header">🎯 Análisis de Desempeño</div>', unsafe_allow_html=True)
    
    # Combinar datos
    performance_data = df_empleados.merge(
        df_asistencias.groupby('empleado_id').agg({
            'productividad': 'mean',
            'calidad_trabajo': 'mean',
            'horas_extra': 'sum',
            'ausente': 'sum'
        }).reset_index(),
        left_on='id', right_on='empleado_id'
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 - Productividad")
        
        top_performers = performance_data.nlargest(10, 'productividad')[
            ['nombre', 'apellido', 'departamento', 'productividad', 'calidad_trabajo']
        ]
        
        for _, emp in top_performers.iterrows():
            st.write(f"**{emp['nombre']} {emp['apellido']}** - {emp['departamento']}")
            st.write(f"Productividad: {emp['productividad']:.1f}% | Calidad: {emp['calidad_trabajo']:.1f}%")
            st.progress(emp['productividad']/100)
            st.markdown("---")
    
    with col2:
        st.subheader("📈 Distribución de Métricas")
        
        if PLOTLY_AVAILABLE:
            # Box plot de productividad por departamento
            fig = create_simple_plotly_chart(
                performance_data,
                'Productividad por Departamento',
                'box',
                x='departamento',
                y='productividad'
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

def show_early_warnings(df_empleados, df_obras, df_asistencias):
    st.markdown('<div class="section-header">⚠️ Sistema de Alertas</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Alertas Críticas")
        
        # Obras en riesgo
        obras_riesgo = df_obras[df_obras['estado'] == 'En Riesgo']
        for _, obra in obras_riesgo.iterrows():
            st.error(f"**{obra['nombre']}** - {obra['ubicacion']}")
        
        # Alta rotación por departamento
        rotacion_depto = df_empleados.groupby('departamento')['activo'].mean()
        for depto, tasa in rotacion_depto.items():
            if tasa < 0.85:  # Menos del 85% de empleados activos
                st.error(f"**Alta rotación en {depto}** - {((1-tasa)*100):.1f}%")
    
    with col2:
        st.subheader("🔔 Alertas Preventivas")
        
        # Baja productividad
        productividad_depto = df_asistencias.merge(df_empleados, left_on='empleado_id', right_on='id')
        productividad_depto = productividad_depto.groupby('departamento')['productividad'].mean()
        
        for depto, prod in productividad_depto.items():
            if prod < 75:
                st.warning(f"**Baja productividad en {depto}** - {prod:.1f}%")
        
        # Alto ausentismo
        ausentismo_data = df_asistencias.groupby('empleado_id')['ausente'].sum()
        alto_ausentismo = ausentismo_data[ausentismo_data > 3]
        
        if len(alto_ausentismo) > 0:
            st.warning(f"**{len(alto_ausentismo)} empleados con alto ausentismo**")

def show_configuration():
    st.markdown('<div class="section-header">⚙️ Configuración del Sistema</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🏢 Departamentos", "📊 Métricas", "👤 Usuarios"])
    
    with tab1:
        st.subheader("Gestión de Departamentos")
        
        deptos = ['Albañilería', 'Electricidad', 'Plomería', 'Herrería', 'Pintura']
        
        for depto in deptos:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{depto}**")
            with col2:
                st.button("Editar", key=f"edit_{depto}")
        
        st.divider()
        
        nuevo_depto = st.text_input("Nuevo departamento")
        if st.button("➕ Agregar Departamento"):
            if nuevo_depto:
                st.success(f"Departamento '{nuevo_depto}' agregado!")
    
    with tab2:
        st.subheader("Configuración de KPIs")
        
        kpis = [
            {"nombre": "Productividad", "meta": 85, "activo": True},
            {"nombre": "Calidad", "meta": 90, "activo": True},
            {"nombre": "Ausentismo", "meta": 3, "activo": True},
            {"nombre": "Rotación", "meta": 8, "activo": True}
        ]
        
        for kpi in kpis:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{kpi['nombre']}**")
            with col2:
                st.number_input("Meta", value=kpi['meta'], key=f"meta_{kpi['nombre']}")
            with col3:
                st.checkbox("Activo", value=kpi['activo'], key=f"act_{kpi['nombre']}")
    
    with tab3:
        st.subheader("Gestión de Usuarios")
        
        usuarios = [
            {"usuario": "admin", "rol": "Administrador", "activo": True},
            {"usuario": "gerente.rrhh", "rol": "Gerente RRHH", "activo": True},
            {"usuario": "supervisor", "rol": "Supervisor", "activo": True}
        ]
        
        for usuario in usuarios:
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**{usuario['usuario']}**")
            with col2:
                st.write(usuario['rol'])
            with col3:
                st.checkbox("Activo", value=usuario['activo'], key=f"user_{usuario['usuario']}")

if __name__ == "__main__":
    main()
