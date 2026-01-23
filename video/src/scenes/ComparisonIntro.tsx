import React from "react";
import {
  AbsoluteFill,
  spring,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
} from "remotion";
import { colors, gradients } from "../styles/colors";
import { fonts, typography } from "../styles/fonts";

export const ComparisonIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const fadeIn = spring({
    frame,
    fps,
    config: { damping: 200 },
  });

  const arrowProgress = spring({
    frame: frame - 15,
    fps,
    config: { damping: 15, stiffness: 150 },
  });

  const arrowTranslate = interpolate(arrowProgress, [0, 1], [-20, 0]);

  return (
    <AbsoluteFill
      style={{
        backgroundColor: colors.background,
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: gradients.subtleVignette,
        }}
      />

      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          gap: 60,
          opacity: fadeIn,
        }}
      >
        {/* Before */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 12,
          }}
        >
          <div
            style={{
              fontFamily: fonts.sans,
              fontSize: typography.title.fontSize,
              fontWeight: 600,
              color: colors.textMuted,
            }}
          >
            Before
          </div>
          <div
            style={{
              width: 80,
              height: 4,
              borderRadius: 2,
              backgroundColor: colors.warningRed,
              opacity: 0.6,
            }}
          />
        </div>

        {/* Arrow */}
        <div
          style={{
            fontSize: 72,
            color: colors.accentBlue,
            transform: `translateX(${arrowTranslate}px)`,
            opacity: Math.max(0, arrowProgress),
            textShadow: `0 0 30px ${colors.accentBlueGlow}`,
          }}
        >
          →
        </div>

        {/* After */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 12,
          }}
        >
          <div
            style={{
              fontFamily: fonts.sans,
              fontSize: typography.title.fontSize,
              fontWeight: 700,
              color: colors.successGreenBright,
              textShadow: `0 0 30px ${colors.successGreenGlow}`,
            }}
          >
            After
          </div>
          <div
            style={{
              width: 80,
              height: 4,
              borderRadius: 2,
              backgroundColor: colors.successGreen,
            }}
          />
        </div>
      </div>
    </AbsoluteFill>
  );
};
