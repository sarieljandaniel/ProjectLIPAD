import { useState } from "react";
import { Sidebar, TabId } from "./components/layout/Sidebar";
import { MainLayout } from "./components/layout/MainLayout";
import { HomePage } from "./pages/HomePage";
import { InspectionPage } from "./pages/InspectionPage";
import { AnalysisPage } from "./pages/AnalysisPage";
import { ReportsPage } from "./pages/ReportsPage";

function App() {
  const [tab, setTab] = useState<TabId>("home");

  return (
    <div className="flex min-h-screen flex-col bg-newsprint md:flex-row">
      <Sidebar active={tab} onSelect={setTab} />
      <MainLayout>
        {tab === "home" && <HomePage />}
        {tab === "inspection" && <InspectionPage />}
        {tab === "analysis" && <AnalysisPage />}
        {tab === "reports" && <ReportsPage />}
      </MainLayout>
    </div>
  );
}

export default App;
