import { vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import App from "./App.jsx";

beforeEach(() => {
  global.fetch = vi.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ sequences: ["ARND", "MKT"] }),
    }),
  );
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: vi.fn(() => Promise.resolve()) },
  });
  URL.createObjectURL = vi.fn(() => "blob:fasta");
  URL.revokeObjectURL = vi.fn();
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
});
afterEach(() => vi.restoreAllMocks());

test("submits the selected composition and sampling settings and renders both sequences", async () => {
  render(<App />);
  fireEvent.change(screen.getByLabelText("Token limit"), {
    target: { value: "32" },
  });
  fireEvent.change(screen.getByLabelText("Sequences", { exact: true }), {
    target: { value: "2" },
  });
  fireEvent.change(
    screen.getByLabelText("Hydrophobic / aromatic", { exact: true }),
    { target: { value: "0.8" } },
  );
  fireEvent.click(
    screen.getByRole("button", { name: "Generate sequences", exact: true }),
  );
  expect(
    await screen.findByRole("button", { name: "Download FASTA", exact: true }),
  ).toBeEnabled();
  expect(screen.getByLabelText("ARND")).toBeInTheDocument();
  expect(screen.getByLabelText("MKT")).toBeInTheDocument();
  const body = JSON.parse(fetch.mock.calls[0][1].body);
  expect(body.max_length).toBe(32);
  expect(body.num_sequences).toBe(2);
  expect(body.properties.thermostable).toBe(0.8);
});

test("copies and exports the generated batch", async () => {
  render(<App />);
  fireEvent.click(
    screen.getByRole("button", { name: "Generate sequences", exact: true }),
  );
  fireEvent.click(
    await screen.findByRole("button", { name: "Copy sequence 1", exact: true }),
  );
  expect(navigator.clipboard.writeText).toHaveBeenCalledWith("ARND");
  expect(await screen.findByText("COPIED TO CLIPBOARD")).toBeInTheDocument();
  fetch.mockResolvedValueOnce({
    ok: true,
    json: () => Promise.resolve({ fasta: ">sequence1\nARND\n" }),
  });
  fireEvent.click(
    screen.getByRole("button", { name: "Download FASTA", exact: true }),
  );
  expect(
    await screen.findByRole("button", { name: "Clear batch", exact: true }),
  ).toBeInTheDocument();
  expect(fetch).toHaveBeenLastCalledWith(
    "/api/export",
    expect.objectContaining({
      body: JSON.stringify({ sequences: ["ARND", "MKT"] }),
    }),
  );
});

test("shows a missing-model error and allows retry", async () => {
  fetch.mockResolvedValueOnce({
    ok: false,
    json: () => Promise.resolve({ detail: "Model bundle unavailable" }),
  });
  render(<App />);
  fireEvent.click(
    screen.getByRole("button", { name: "Generate sequences", exact: true }),
  );
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Model bundle unavailable",
  );
  expect(
    screen.getByRole("button", { name: "Generate sequences", exact: true }),
  ).toBeEnabled();
});
