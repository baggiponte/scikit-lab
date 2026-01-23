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

const PainPoint: React.FC<{
  text: string;
  startFrame: number;
  index: number;
}> = ({ text, startFrame, index }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const relativeFrame = frame - startFrame;

  if (relativeFrame < 0) return null;

  const opacity = spring({
    frame: relativeFrame,
    fps,
    config: { damping: 200 },
  });

  const translateX = spring({
    frame: relativeFrame,
    fps,
    config: { damping: 20, stiffness: 150 },
    from: -80,
    to: 0,
  });

  // Shake animation for the X
  const shake = relativeFrame > 10 && relativeFrame < 25
    ? interpolate(
        Math.sin(relativeFrame * 1.5),
        [-1, 1],
        [-3, 3]
      )
    : 0;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 28,
        opacity,
        transform: `translateX(${translateX}px)`,
        marginBottom: 40,
      }}
    >
      <div
        style={{
          width: 56,
          height: 56,
          borderRadius: 12,
          backgroundColor: "rgba(239, 68, 68, 0.15)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          transform: `rotate(${shake}deg)`,
          border: `2px solid ${colors.warningRed}`,
        }}
      >
        <span
          style={{
            color: colors.warningRedBright,
            fontSize: 32,
            fontWeight: 700,
          }}
        >
          ✕
        </span>
      </div>
      <span
        style={{
          fontFamily: fonts.sans,
          fontSize: typography.subtitle.fontSize,
          fontWeight: 500,
          color: colors.text,
          letterSpacing: "-0.01em",
        }}
      >
        {text}
      </span>
    </div>
  );
};

export const ProblemStatement: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = spring({
    frame,
    fps,
    config: { damping: 200 },
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: colors.background,
      }}
    >
      {/* Subtle red glow */}
      <div
        style={{
          position: "absolute",
          top: "30%",
          left: "10%",
          width: 400,
          height: 400,
          background: "radial-gradient(circle, rgba(239, 68, 68, 0.1) 0%, transparent 70%)",
          filter: "blur(60px)",
        }}
      />

      <div
        style={{
          position: "absolute",
          inset: 0,
          background: gradients.subtleVignette,
        }}
      />

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          height: "100%",
          paddingLeft: 200,
          paddingRight: 100,
        }}
      >
        <div
          style={{
            fontFamily: fonts.sans,
            fontSize: 28,
            fontWeight: 600,
            color: colors.warningRed,
            marginBottom: 48,
            opacity: titleOpacity,
            textTransform: "uppercase",
            letterSpacing: "0.1em",
          }}
        >
          The Problem
        </div>

        <PainPoint text="Boilerplate everywhere" startFrame={15} index={0} />
        <PainPoint text="Metrics scattered across files" startFrame={35} index={1} />
        <PainPoint text="Every project reinvents logging" startFrame={55} index={2} />
      </div>
    </AbsoluteFill>
  );
};
