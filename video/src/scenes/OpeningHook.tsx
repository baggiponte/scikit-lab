import React from "react";
import {
  AbsoluteFill,
  spring,
  useCurrentFrame,
  useVideoConfig,
  Easing,
  interpolate,
} from "remotion";
import { TypewriterText } from "../components/TypewriterText";
import { colors, gradients } from "../styles/colors";
import { fonts, typography } from "../styles/fonts";

export const OpeningHook: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const fadeIn = spring({
    frame,
    fps,
    config: { damping: 200 },
  });

  // Subtle background pulse
  const glowIntensity = interpolate(
    Math.sin(frame * 0.05),
    [-1, 1],
    [0.3, 0.5]
  );

  return (
    <AbsoluteFill
      style={{
        backgroundColor: colors.background,
      }}
    >
      {/* Animated gradient background */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: gradients.radialGlow,
          opacity: glowIntensity * fadeIn,
        }}
      />

      {/* Vignette */}
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
          justifyContent: "center",
          alignItems: "center",
          opacity: fadeIn,
        }}
      >
        <div
          style={{
            textAlign: "center",
            maxWidth: 1200,
            padding: 40,
          }}
        >
          <div style={{ marginBottom: 16 }}>
            <TypewriterText
              text="What if sklearn experiments..."
              startFrame={10}
              framesPerChar={2}
              fontSize={typography.hero.fontSize}
              fontWeight={500}
              color={colors.text}
              fontFamily={fonts.sans}
            />
          </div>
          <TypewriterText
            text="just worked?"
            startFrame={70}
            framesPerChar={3}
            fontSize={80}
            fontWeight={700}
            color={colors.accentBlueBright}
            fontFamily={fonts.sans}
            style={{
              textShadow: `0 0 60px ${colors.accentBlueGlow}`,
            }}
          />
        </div>
      </div>
    </AbsoluteFill>
  );
};
