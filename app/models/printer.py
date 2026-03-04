from pydantic import BaseModel


class PrinterProfile(BaseModel):
    id: str
    name: str
    plate_width_mm: float
    plate_height_mm: float

