"""
Esquemas Pydantic para la API de inferencia
Define la estructura de entrada y salida de la API.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from enum import Enum

# === Enums para campos categóricos ===
class ParentalInvolvement(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class AccessToResources(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class MotivationLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class FamilyIncome(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class TeacherQuality(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class SchoolType(str, Enum):
    PUBLIC = "Public"
    PRIVATE = "Private"

class PeerInfluence(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"

class DistanceFromHome(str, Enum):
    NEAR = "Near"
    MODERATE = "Moderate"
    FAR = "Far"

class GenderType(str, Enum):
    MALE = "Male"
    FEMALE = "Female"

class ParentalEducationLevel(str, Enum):
    HIGH_SCHOOL = "High School"
    COLLEGE = "College"
    POSTGRADUATE = "Postgraduate"

# === Schema de entrada ===
class StudentInput(BaseModel):
    """Schema de entrada para predicción de rendimiento estudiantil."""
    Hours_Studied: float = Field(..., ge=0, le=50, description="Horas de estudio por semana")
    Attendance: float = Field(..., ge=0, le=100, description="Porcentaje de asistencia")
    Sleep_Hours: float = Field(..., ge=0, le=12, description="Horas de sueño por noche")
    Previous_Scores: float = Field(..., ge=0, le=100, description="Puntuaciones anteriores")
    Tutoring_Sessions: int = Field(..., ge=0, le=20, description="Sesiones de tutoría")
    Physical_Activity: int = Field(..., ge=0, le=10, description="Horas de actividad física/semana")
    
    Parental_Involvement: ParentalInvolvement = Field(..., description="Involucramiento parental")
    Access_to_Resources: AccessToResources = Field(..., description="Acceso a recursos educativos")
    Extracurricular_Activities: str = Field(..., description="Actividades extracurriculares (Yes/No)")
    Motivation_Level: MotivationLevel = Field(..., description="Nivel de motivación")
    Internet_Access: str = Field(..., description="Acceso a internet (Yes/No)")
    Family_Income: FamilyIncome = Field(..., description="Ingreso familiar")
    Teacher_Quality: TeacherQuality = Field(..., description="Calidad del profesor")
    School_Type: SchoolType = Field(..., description="Tipo de escuela")
    Peer_Influence: PeerInfluence = Field(..., description="Influencia de compañeros")
    Learning_Disabilities: str = Field(..., description="Discapacidades de aprendizaje (Yes/No)")
    Parental_Education_Level: ParentalEducationLevel = Field(..., description="Nivel educativo de padres")
    Distance_from_Home: DistanceFromHome = Field(..., description="Distancia desde casa")
    Gender: GenderType = Field(..., description="Género")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Hours_Studied": 20, "Attendance": 85, "Sleep_Hours": 7, "Previous_Scores": 75,
                "Tutoring_Sessions": 2, "Physical_Activity": 3, "Parental_Involvement": "Medium",
                "Access_to_Resources": "Medium", "Extracurricular_Activities": "Yes",
                "Motivation_Level": "Medium", "Internet_Access": "Yes", "Family_Income": "Medium",
                "Teacher_Quality": "Medium", "School_Type": "Public", "Peer_Influence": "Positive",
                "Learning_Disabilities": "No", "Parental_Education_Level": "College",
                "Distance_from_Home": "Near", "Gender": "Male"
            }
        }
    )

# === Schema de salida ===
class PredictionOutput(BaseModel):
    """Schema de salida de la predicción."""
    prediction: float = Field(..., description="Puntuación predicha del examen")
    prediction_rounded: int = Field(..., description="Puntuación predicha redondeada")
    model_version: str = Field(..., description="Versión del modelo usado")
    model_name: str = Field(..., description="Nombre del modelo")
    
    # ⭐ ESTA ES LA LÍNEA QUE FALTABA Y CAUSABA EL PROBLEMA
    pod_name: str = Field(..., description="Nombre del pod que atendió la petición")
    
    confidence_note: str = Field(
        default="Prediction based on trained model",
        description="Nota sobre la predicción"
    )
    
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "prediction": 72.45,
                "prediction_rounded": 72,
                "model_version": "1",
                "model_name": "student-performance-model",
                "pod_name": "student-performance-api-xxxxx",
                "confidence_note": "Prediction based on trained model"
            }
        }
    )

class HealthCheck(BaseModel):
    """Schema para health check de la API."""
    status: str = "healthy"
    model_loaded: bool = True
    model_version: Optional[str] = None
    model_name: Optional[str] = None
    
    # ⭐ TAMBIÉN AGREGADO AQUÍ PARA CONSISTENCIA
    pod_name: str = "unknown"
    
    model_config = ConfigDict(
        protected_namespaces=()
    )