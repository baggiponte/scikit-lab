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

const MethodItem: React.FC<{
  method: string;
  description: string;
  startFrame: number;
  index: number;
}> = ({ method, description, startFrame, index }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const relativeFrame = frame - startFrame;

  if (relativeFrame < 0) return null;

  const opacity = spring({
    frame: relativeFrame,
    fps,
    config: { damping: 200 },
  });

  const translateY = spring({
    frame: relativeFrame,
    fps,
    config: { damping: 18, stiffness: 120 },
    from: 30,
    to: 0,
  });

  const scale = spring({
    frame: relativeFrame,
    fps,
    config: { damping: 200 },
    from: 0.95,
    to: 1,
  });

  // Staggered glow intensity
  const glowDelay = index * 15;
  const glowIntensity = relativeFrame > glowDelay
    ? interpolate(
        Math.sin((relativeFrame - glowDelay) * 0.1),
        [-1, 1],
        [0.3, 0.6]
      )
    : 0.3;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 24,
        opacity,
        transform: `translateY(${translateY}px) scale(${scale})`,
        marginBottom: 28,
        padding: "16px 24px",
        borderRadius: 12,
        backgroundColor: "rgba(59, 130, 246, 0.05)",
        border: `1px solid rgba(59, 130, 246, ${glowIntensity * 0.3})`,
      }}
    >
      <span
        style={{
          fontFamily: fonts.mono,
          fontSize: 32,
          fontWeight: 600,
          color: colors.accentBlueBright,
          textShadow: `0 0 20px rgba(59, 130, 246, ${glowIntensity})`,
        }}
      >
        experiment.
        <span style={{ color: colors.codeYellow }}>{method}</span>
        <span style={{ color: colors.textMuted }}>()</span>
      </span>
      <span
        style={{
          fontFamily: fonts.sans,
          fontSize: typography.body.fontSize,
          fontWeight: 400,
          color: colors.textMuted,
          marginLeft: 8,
        }}
      >
        {description}
      </span>
    </div>
  );
};

export const ThePattern: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = spring({
    frame,
    fps,
    config: { damping: 200 },
  });

  const titleScale = spring({
    frame,
    fps,
    config: { damping: 200 },
    from: 0.95,
    to: 1,
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: colors.background,
      }}
    >
      {/* Background glow */}
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: 800,
          height: 600,
          background: gradients.radialGlow,
          opacity: 0.4,
        }}
      />

      <div
        style={{
          position: "absolute",
          inset: 0,
          background: gradients.subtleVignette,
        }}
      />

      {/* Content */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          padding: 60,
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-start",
          }}
        >
          {/* Title */}
          <div
            style={{
              fontFamily: fonts.sans,
              fontSize: typography.title.fontSize,
              fontWeight: 700,
              color: colors.text,
              marginBottom: 48,
              opacity: titleOpacity,
              transform: `scale(${titleScale})`,
              letterSpacing: typography.title.letterSpacing,
            }}
          >
            One unified interface
          </div>

          {/* Methods */}
          <MethodItem method="fit" description="Train & log" startFrame={20} index={0} />
          <MethodItem method="evaluate" description="Metrics & plots" startFrame={40} index={1} />
          <MethodItem method="cross_validate" description="Per-fold everything" startFrame={60} index={2} />
          <MethodItem method="search" description="All trials logged" startFrame={80} index={3} />
        </div>
      </div>
    </AbsoluteFill>
  );
};
