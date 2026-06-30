/**
 * Glucose chart: fixed y-axis (mg/dL) + horizontally-scrollable plot with an
 * hourly grid, shaded normal-range band, raw readings (solid) and predicted
 * readings (dashed). Auto-scrolls to "now". Pure react-native-svg.
 */
import { useEffect, useRef } from 'react';
import { ScrollView, View } from 'react-native';
import Svg, { Circle, G, Line, Path, Rect, Text as SvgText } from 'react-native-svg';
import { useColors } from '@/lib/theme';
import type { GlucosePoint } from '@/lib/mock';

type Props = {
  raw: GlucosePoint[];
  predicted: GlucosePoint[];
  now: number;
  range: { low: number; high: number };
  foodMarkers?: { t: number }[];
  height?: number;
  width?: number;
};

const Y_MIN = 40;
const Y_MAX = 260;
const Y_TICKS = [40, 80, 120, 160, 200, 240];
const TOP = 12;
const BOTTOM = 22;     // room for hour labels
const LEFT = 10;
const RIGHT = 16;
const HOUR_W = 74;     // px per hour
const YAXIS_W = 32;
const HOUR_MS = 3_600_000;

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const fmtHour = (t: number) => {
  const h = new Date(t).getHours();
  return `${(h % 12) || 12} ${h >= 12 ? 'PM' : 'AM'}`;
};

export function GlucoseChart({ raw, predicted, now, range, foodMarkers, height = 220, width = 340 }: Props) {
  const c = useColors();
  const scrollRef = useRef<ScrollView>(null);

  const all = [...raw, ...predicted];
  const tMin = all[0]?.t ?? now;
  const tMax = all[all.length - 1]?.t ?? now + HOUR_MS;

  // glucose value at an arbitrary time (linear-interpolated) → place food icons on the line
  const sorted = all.filter((p) => Number.isFinite(p.t)).sort((a, b) => a.t - b.t);
  const glucoseAt = (t: number) => {
    if (!sorted.length) return Y_MIN;
    if (t <= sorted[0].t) return sorted[0].mgdl;
    if (t >= sorted[sorted.length - 1].t) return sorted[sorted.length - 1].mgdl;
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].t >= t) {
        const a = sorted[i - 1], b = sorted[i];
        return a.mgdl + ((t - a.t) / ((b.t - a.t) || 1)) * (b.mgdl - a.mgdl);
      }
    }
    return sorted[sorted.length - 1].mgdl;
  };
  const marks = (foodMarkers ?? []).filter((m) => m.t >= tMin && m.t <= tMax);
  const spanMin = Math.max(60, (tMax - tMin) / 60000);
  const viewportW = Math.max(0, width - YAXIS_W);
  const contentW = Math.max(viewportW, (spanMin / 60) * HOUR_W);
  const innerH = height - TOP - BOTTOM;
  const innerW = contentW - LEFT - RIGHT;

  const sx = (t: number) => LEFT + ((t - tMin) / Math.max(1, tMax - tMin)) * innerW;
  const sy = (g: number) => TOP + (1 - (clamp(g, Y_MIN, Y_MAX) - Y_MIN) / (Y_MAX - Y_MIN)) * innerH;
  const line = (pts: GlucosePoint[]) =>
    pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p.t).toFixed(1)},${sy(p.mgdl).toFixed(1)}`).join(' ');

  const hourTicks: number[] = [];
  for (let t = Math.ceil(tMin / HOUR_MS) * HOUR_MS; t <= tMax; t += HOUR_MS) hourTicks.push(t);

  // auto-scroll so "now" sits ~62% across the viewport (recent past + forecast visible)
  useEffect(() => {
    const x = Math.max(0, sx(now) - viewportW * 0.62);
    const id = setTimeout(() => scrollRef.current?.scrollTo({ x, animated: false }), 0);
    return () => clearTimeout(id);
  }, [contentW, now, viewportW]);   // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <View style={{ flexDirection: 'row' }}>
      {/* fixed y-axis */}
      <Svg width={YAXIS_W} height={height}>
        {Y_TICKS.map((v) => (
          <SvgText key={v} x={YAXIS_W - 5} y={sy(v) + 3} fontSize="10" fill={c.textMuted} textAnchor="end">{v}</SvgText>
        ))}
      </Svg>

      {/* scrollable plot */}
      <ScrollView ref={scrollRef} horizontal showsHorizontalScrollIndicator={false}>
        <Svg width={contentW} height={height}>
          {/* horizontal gridlines */}
          {Y_TICKS.map((v) => (
            <Line key={`h${v}`} x1={0} x2={contentW} y1={sy(v)} y2={sy(v)} stroke={c.border} strokeOpacity={0.45} strokeWidth={1} />
          ))}

          {/* normal-range band + boundary lines */}
          <Rect x={0} y={sy(range.high)} width={contentW} height={Math.max(0, sy(range.low) - sy(range.high))} fill={c.inRangeBand} />
          <Line x1={0} x2={contentW} y1={sy(range.high)} y2={sy(range.high)} stroke={c.inRange} strokeOpacity={0.4} strokeDasharray="3 4" />
          <Line x1={0} x2={contentW} y1={sy(range.low)} y2={sy(range.low)} stroke={c.inRange} strokeOpacity={0.4} strokeDasharray="3 4" />

          {/* vertical hour gridlines */}
          {hourTicks.map((t) => (
            <Line key={`v${t}`} x1={sx(t)} x2={sx(t)} y1={TOP} y2={height - BOTTOM} stroke={c.border} strokeOpacity={0.35} strokeWidth={1} />
          ))}
          {/* hour labels */}
          {hourTicks.map((t) => (
            <SvgText key={`l${t}`} x={sx(t)} y={height - 6} fontSize="9" fill={c.textMuted} textAnchor="middle">{fmtHour(t)}</SvgText>
          ))}

          {/* "now" marker */}
          <Line x1={sx(now)} x2={sx(now)} y1={TOP} y2={height - BOTTOM} stroke={c.accent} strokeWidth={1.5} strokeDasharray="2 3" />
          <SvgText x={sx(now)} y={TOP + 8} fontSize="9" fill={c.accent} textAnchor="middle">now</SvgText>

          {/* raw (solid) + predicted (dashed) */}
          <Path d={line(raw)} stroke={c.text} strokeWidth={2} fill="none" />
          <Path d={line(predicted)} stroke={c.accent} strokeWidth={2.5} fill="none" strokeDasharray="5 4" />

          {/* food eaten — icon sits on the glucose line at the logged time */}
          {marks.map((m, i) => {
            const x = sx(m.t);
            const y = sy(glucoseAt(m.t));
            return (
              <G key={`food${i}`}>
                <Circle cx={x} cy={y} r={10} fill={c.surface} stroke={c.accent} strokeWidth={1.5} />
                <SvgText x={x} y={y + 4} fontSize={11} textAnchor="middle">🍴</SvgText>
              </G>
            );
          })}
        </Svg>
      </ScrollView>
    </View>
  );
}
