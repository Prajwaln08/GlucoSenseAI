/**
 * Glucose chart: shaded normal-range band, raw readings (solid), and the
 * model's predicted readings (dashed, into the future), with a "now" marker.
 * Pure react-native-svg — no heavy charting dependency.
 */
import { useMemo } from 'react';
import { View } from 'react-native';
import Svg, { Line, Path, Rect, Text as SvgText } from 'react-native-svg';
import { useColors } from '@/lib/theme';
import type { GlucosePoint } from '@/lib/mock';

type Props = {
  raw: GlucosePoint[];
  predicted: GlucosePoint[];
  now: number;
  range: { low: number; high: number };
  height?: number;
  width?: number;
};

const Y_MIN = 40;
const Y_MAX = 260;
const PAD = { top: 10, right: 10, bottom: 18, left: 30 };

export function GlucoseChart({ raw, predicted, now, range, height = 220, width = 340 }: Props) {
  const c = useColors();
  const all = [...raw, ...predicted];

  const { plot } = useMemo(() => {
    const w = width - PAD.left - PAD.right;
    const h = height - PAD.top - PAD.bottom;
    const tMin = all[0]?.t ?? now;
    const tMax = all[all.length - 1]?.t ?? now;
    const sx = (t: number) => PAD.left + ((t - tMin) / Math.max(1, tMax - tMin)) * w;
    const sy = (g: number) => PAD.top + (1 - (clamp(g, Y_MIN, Y_MAX) - Y_MIN) / (Y_MAX - Y_MIN)) * h;
    const line = (pts: GlucosePoint[]) =>
      pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p.t).toFixed(1)},${sy(p.mgdl).toFixed(1)}`).join(' ');
    return {
      plot: {
        w, h, sx, sy,
        rawPath: line(raw),
        predPath: line(predicted),
        bandTop: sy(range.high),
        bandBottom: sy(range.low),
        nowX: sx(now),
      },
    };
  }, [raw, predicted, now, range, width, height]);

  return (
    <View>
      <Svg width={width} height={height}>
        {/* normal-range shaded band (70–180) */}
        <Rect
          x={PAD.left} y={plot.bandTop}
          width={plot.w} height={Math.max(0, plot.bandBottom - plot.bandTop)}
          fill={c.inRangeBand}
        />
        <Line x1={PAD.left} x2={width - PAD.right} y1={plot.bandTop} y2={plot.bandTop}
          stroke={c.inRange} strokeOpacity={0.4} strokeDasharray="3 4" />
        <Line x1={PAD.left} x2={width - PAD.right} y1={plot.bandBottom} y2={plot.bandBottom}
          stroke={c.inRange} strokeOpacity={0.4} strokeDasharray="3 4" />

        {/* y labels */}
        <SvgText x={4} y={plot.bandTop + 4} fontSize="10" fill={c.textMuted}>{range.high}</SvgText>
        <SvgText x={4} y={plot.bandBottom + 4} fontSize="10" fill={c.textMuted}>{range.low}</SvgText>

        {/* "now" divider between raw and predicted */}
        <Line x1={plot.nowX} x2={plot.nowX} y1={PAD.top} y2={height - PAD.bottom}
          stroke={c.border} strokeWidth={1} />
        <SvgText x={plot.nowX - 8} y={height - 5} fontSize="10" fill={c.textMuted}>now</SvgText>

        {/* raw (solid) + predicted (dashed) */}
        <Path d={plot.rawPath} stroke={c.text} strokeWidth={2} fill="none" />
        <Path d={plot.predPath} stroke={c.accent} strokeWidth={2.5} fill="none" strokeDasharray="5 4" />
      </Svg>
    </View>
  );
}

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}
