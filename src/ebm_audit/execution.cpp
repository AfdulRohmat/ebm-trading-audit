#include <cmath>
#include <limits>
#include <algorithm>
// Exact execution cache accelerator; independently compared with v3_runner.
// q: bid OHLC then ask OHLC per row. o: result scalars. path: marked PnL.
extern "C" void trade_path(const double* q, int n, int start_minute, int side,
                           int runner, int opposite, double stress, double entry,
                           double slip, double fee, double distance, double risk,
                           int clock, double activation, double trail_distance,
                           double* o, double* path, double* trace) {
    double stop = entry - side * distance;
    double best = -std::numeric_limits<double>::infinity();
    double exit_quote = 0.;
    int active = 0, updates = 0, plen = 0, reason = -1, exit_minute = -1;
    for (int k = 0; k < n; ++k) {
        double v[4];
        for (int f = 0; f < 4; ++f) {
            double b = q[k * 8 + f], a = q[k * 8 + 4 + f];
            v[f] = (b + a) / 2.0 - side * (stress * (a - b) / 2.0);
        }
        double op = v[0], cl = v[3];
        double adverse = side == 1 ? v[2] : v[1], favorable = side == 1 ? v[1] : v[2];
        if (side * (op - stop) <= 0) {
            exit_quote = op; exit_minute = k; reason = active ? 1 : 0; break;
        }
        if (!runner && k >= 60) { exit_quote = op; exit_minute = k; reason = 2; break; }
        if (runner && opposite >= 0 && k >= opposite) {
            exit_quote = op; exit_minute = k; reason = 3; break;
        }
        path[plen++] = side * (op - entry) - slip - fee;
        if (side * (adverse - stop) <= 0) {
            exit_quote = stop; exit_minute = k; reason = active ? 1 : 0; break;
        }
        for (double price : {favorable, adverse, cl}) path[plen++] = side * (price - entry) - slip - fee;
        if (k == n - 1) { exit_quote = cl; exit_minute = k + 1; reason = 4; break; }
        if (runner && (start_minute + k + 1) % clock == 0) {
            double net_r = (side * (cl - entry) - slip - fee) / risk;
            best = std::max(best, net_r);
            if (best >= activation) {
                double locked = best - trail_distance;
                double candidate = entry + side * (locked * risk + slip + fee);
                if (side * (candidate - stop) > 0) {
                    trace[updates * 5] = k + 1;
                    trace[updates * 5 + 1] = stop;
                    trace[updates * 5 + 2] = candidate;
                    trace[updates * 5 + 3] = best;
                    trace[updates * 5 + 4] = locked;
                    ++updates;
                    stop = candidate; active = 1;
                }
            }
        }
    }
    double exit_price = exit_quote - side * slip;
    double net = side * (exit_price - entry) - fee;
    path[plen++] = net;
    o[0] = exit_price; o[1] = stop; o[2] = net; o[3] = exit_minute;
    o[4] = reason; o[5] = active; o[6] = updates; o[7] = plen;
}
