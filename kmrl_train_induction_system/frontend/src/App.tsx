import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "@/components/theme-provider";
import ErrorBoundary from "@/components/ErrorBoundary";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import Dashboard from "./pages/Dashboard";
import Assignments from "./pages/Assignments";
import Reports from "./pages/Reports";
import Settings from "./pages/Settings";
import Trainsets from "./pages/Trainsets";
import Optimization from "./pages/Optimization";
import DataIngestion from "./pages/DataIngestion";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="system" storageKey="kmrl-ui-theme">
        <BrowserRouter>
          <DashboardLayout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/assignments" element={<Assignments />} />
              <Route path="/trainsets" element={<Trainsets />} />
              <Route path="/optimization" element={<Optimization />} />
              <Route path="/data-ingestion" element={<DataIngestion />} />
              <Route path="/reports" element={<Reports />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </DashboardLayout>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  </ErrorBoundary>
);

export default App;