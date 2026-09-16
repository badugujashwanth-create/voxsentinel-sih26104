import { useState } from "react";
import { ScenarioLauncher } from "../components/demo/ScenarioLauncher";
import { SecurityConsole } from "../components/console/SecurityConsole";
import { useDemoController } from "../hooks/useDemoController";

/** Renders the launcher and the single evolving VoxSentinel demo experience. */
export function App() {
  const controller = useDemoController();
  const [verificationOpen, setVerificationOpen] = useState(false);

  if (!controller.selectedScenario || !controller.context) {
    return <ScenarioLauncher onSelect={controller.selectScenario} />;
  }

  return <SecurityConsole controller={controller} verificationOpen={verificationOpen} onOpenVerification={() => setVerificationOpen(true)} onCloseVerification={() => setVerificationOpen(false)} />;
}
