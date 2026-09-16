export const VERIFICATION_METHODS = ["OTP", "VOICE_CHALLENGE", "VERIFIED_CALLBACK", "SUPERVISOR_APPROVAL"] as const;
export type VerificationMethod = (typeof VERIFICATION_METHODS)[number];
export type VerificationStatus = "IDLE" | "PENDING" | "VERIFIED" | "FAILED";

export interface VerificationState {
  method: VerificationMethod | null;
  status: VerificationStatus;
  step: string;
  voiceRiskPreserved: boolean;
}

export type VerificationAction =
  | { type: "REQUEST"; method: VerificationMethod }
  | { type: "SENT" }
  | { type: "START" }
  | { type: "VERIFY" }
  | { type: "FAIL" }
  | { type: "RESET" };

export const initialVerificationState: VerificationState = {
  method: null,
  status: "IDLE",
  step: "",
  voiceRiskPreserved: true,
};

/** Reduces lightweight verification actions without mutating voice-risk state. */
export function verificationReducer(state: VerificationState, action: VerificationAction): VerificationState {
  switch (action.type) {
    case "REQUEST":
      return { ...state, method: action.method, status: "PENDING", step: getRequestedStep(action.method) };
    case "SENT":
      return { ...state, status: "PENDING", step: "Verification pending" };
    case "START":
      return { ...state, status: "PENDING", step: getInProgressStep(state.method) };
    case "VERIFY":
      return { ...state, status: "VERIFIED", step: "Identity confirmed" };
    case "FAIL":
      return { ...state, status: "FAILED", step: "Verification failed" };
    case "RESET":
      return initialVerificationState;
  }
}

/** Maps a verification method to its initial operator-facing step. */
function getRequestedStep(method: VerificationMethod): string {
  if (method === "VERIFIED_CALLBACK") return "Callback initiated";
  if (method === "VOICE_CHALLENGE") return "Challenge phrase ready";
  if (method === "SUPERVISOR_APPROVAL") return "Awaiting supervisor";
  return "OTP requested";
}

/** Maps a verification method to its in-progress operator-facing step. */
function getInProgressStep(method: VerificationMethod | null): string {
  if (method === "VERIFIED_CALLBACK") return "Secure channel established";
  if (method === "VOICE_CHALLENGE") return "Challenge response pending";
  if (method === "SUPERVISOR_APPROVAL") return "Supervisor decision pending";
  return "OTP verification pending";
}
