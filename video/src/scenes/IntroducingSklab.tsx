import React from "react";
import {
  AbsoluteFill,
  spring,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
} from "remotion";
import { TypewriterText } from "../components/TypewriterText";
import { colors, gradients } from "../styles/colors";
import { fonts, typography } from "../styles/fonts";

export const IntroducingSklab: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Logo entrance with bounce
  const logoScale = spring({
    frame,
    fps,
    config: { damping: 10, stiffness: 100 },
  });

  const logoOpacity = spring({
    frame,
    fps,
    config: { damping: 200 },
  });

  const taglineOpacity = spring({
    frame: frame - 35,
    fps,
    config: { damping: 200 },
  });

  // Glow pulse
  const glowIntensity = interpolate(
    Math.sin(frame * 0.08),
    [-1, 1],
    [0.4, 0.7]
  );

  return (
    <AbsoluteFill
      style={{
        backgroundColor: colors.background,
      }}
    >
      {/* Animated glow behind logo */}
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: 600,
          height: 300,
          background: gradients.radialGlow,
          opacity: glowIntensity * logoOpacity,
          filter: "blur(40px)",
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
          gap: 40,
        }}
      >
        {/* Logo */}
        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 140,
            fontWeight: 700,
            color: colors.accentBlueBright,
            transform: `scale(${logoScale})`,
            opacity: logoOpacity,
            textShadow: `0 0 80px ${colors.accentBlueGlow}, 0 0 120px ${colors.accentBlueGlow}`,
            letterSpacing: "-0.02em",
          }}
        >
          sklab
        </div>

        {/* Tagline */}
        <div
          style={{
            opacity: Math.max(0, taglineOpacity),
          }}
        >
          <TypewriterText
            text="Zero-boilerplate sklearn experiments"
            startFrame={45}
            framesPerChar={2}
            fontSize={typography.subtitle.fontSize}
            fontWeight={400}
            color={colors.textMuted}
            fontFamily={fonts.sans}
            showCursor={true}
          />
        </div>
      </div>
    </AbsoluteFill>
  );
};
