import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";

afterEach(() => vi.unstubAllGlobals());

describe("typed API client", () => {
  it("unwraps the backend data envelope and constructs supported filters", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ data: { items: [], page: 1, page_size: 10, total: 0, has_next: false } }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const data = await api.transactions({ page: 1, page_size: 10, predicted: true, type: "TRANSFER" });
    expect(data.total).toBe(0);
    expect(fetchMock.mock.calls[0]?.[0]).toContain("predicted=true");
    expect(fetchMock.mock.calls[0]?.[0]).toContain("type=TRANSFER");
  });

  it("surfaces API error codes and messages instead of fake fallback data", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { code: "transaction_not_found", message: "Transaction was not found." } }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      }),
    ));
    await expect(api.transaction("missing")).rejects.toMatchObject({
      status: 404,
      code: "transaction_not_found",
      message: "Transaction was not found.",
    });
  });

  it("reports network failures with an actionable backend hint", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fetch failed")));
    await expect(api.health()).rejects.toThrow("Start the backend");
  });
});
