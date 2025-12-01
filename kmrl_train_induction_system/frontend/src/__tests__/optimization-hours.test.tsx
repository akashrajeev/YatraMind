import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Optimization from "../pages/Optimization";
import { optimizationApi } from "../services/api";

jest.mock("../services/api", () => {
  const original = jest.requireActual("../services/api");
  return {
    ...original,
    optimizationApi: {
      ...original.optimizationApi,
      runOptimization: jest.fn(),
      getStablingGeometry: jest.fn(),
      getShuntingSchedule: jest.fn(),
    },
  };
});

const mockedOptimizationApi = optimizationApi as jest.Mocked<typeof optimizationApi>;

const renderWithClient = () => {
  const client = new QueryClient();
  return render(
    <QueryClientProvider client={client}>
      <Optimization />
    </QueryClientProvider>
  );
};

describe("Optimization Required Service Hours flow", () => {
  beforeEach(() => {
    jest.resetAllMocks();
  });

  it("shows diagnostic note and refreshes stabling/shunting after optimization run", async () => {
    mockedOptimizationApi.runOptimization.mockResolvedValueOnce({
      data: {
        requested_service_hours: 4,
        requested_train_count: 1,
        eligible_train_count: 1,
        granted_train_count: 1,
        note: "Requested 4.0 service hours → approximately 1 trains (avg 12.0 hrs/train).",
        decisions: [],
        stabling_geometry: {},
      },
    } as any);

    mockedOptimizationApi.getStablingGeometry.mockResolvedValueOnce({ data: { total_optimized_positions: 0 } } as any);
    mockedOptimizationApi.getShuntingSchedule.mockResolvedValueOnce({
      data: { total_operations: 0, estimated_total_time: 0, crew_requirements: {} },
    } as any);

    renderWithClient();

    const input = await screen.findByLabelText(/Required Service Hours/i);
    fireEvent.change(input, { target: { value: "4" } });

    const button = screen.getByText(/Run AI\/ML Optimization/i);
    fireEvent.click(button);

    await waitFor(() => {
      expect(mockedOptimizationApi.runOptimization).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.getByText(/Requested 4.0 service hours/i)).toBeInTheDocument();
    });

    expect(mockedOptimizationApi.getStablingGeometry).toHaveBeenCalled();
    expect(mockedOptimizationApi.getShuntingSchedule).toHaveBeenCalled();
  });
});
