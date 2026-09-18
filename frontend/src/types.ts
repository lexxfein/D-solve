export type Treatment = "whitening" | "alignment" | "veneers" | "combined";
export type DesignStatus = "Draft" | "Processing" | "Completed";

export interface Patient {
  id: string;
  name: string;
  age: number;
  email: string;
  phone: string;
  notes: string;
  createdAt: string;
}

export interface SmileDesign {
  id: string;
  patientId: string;
  patientName: string;
  treatment: Treatment;
  intensity: number;
  status: DesignStatus;
  imageData?: string;
  resultData?: string;
  createdAt: string;
  updatedAt: string;
}
