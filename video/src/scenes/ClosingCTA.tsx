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

export const ClosingCTA: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const logoOpacity = spring({
    frame,
    fps,
    config: { damping: 200 },
  });

  const logoScale = spring({
    frame,
    fps,
    config: { damping: 12, stiffness: 100 },
  });

  const installOpacity = spring({
    frame: frame - 30,
    fps,
    config: { damping: 200 },
  });

  const installScale = spring({
    frame: frame - 30,
    fps,
    config: { damping: 20, stiffness: 150 },
    from: 0.95,
    to: 1,
  });

  const linkOpacity = spring({
    frame: frame - 100,
    fps,
    config: { damping: 200 },
  });

  // Glow pulse
  const glowIntensity = interpolate(
    Math.sin(frame * 0.06),
    [-1, 1],
    [0.4, 0.7]
  );

  return (
    <AbsoluteFill
      style={{
        backgroundColor: colors.background,
      }}
    >
      {/* Animated glow */}
      <div
        style={{
          position: "absolute",
          top: "35%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: 600,
          height: 300,
          background: gradients.radialGlow,
          opacity: glowIntensity * logoOpacity,
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

      {/* Content */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          gap: 48,
        }}
      >
        {/* Logo */}
        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 100,
            fontWeight: 700,
            color: colors.accentBlueBright,
            opacity: logoOpacity,
            transform: `scale(${logoScale})`,
            textShadow: `0 0 60px ${colors.accentBlueGlow}, 0 0 100px ${colors.accentBlueGlow}`,
            letterSpacing: "-0.02em",
          }}
        >
          sklab
        </div>

        {/* Install command */}
        <div
          style={{
            opacity: Math.max(0, installOpacity),
            transform: `scale(${installScale})`,
          }}
        >
          <div
            style={{
              position: "relative",
            }}
          >
            {/* Glow behind command box */}
            <div
              style={{
                position: "absolute",
                inset: -12,
                background: colors.successGreenGlow,
                borderRadius: 24,
                filter: "blur(20px)",
                opacity: 0.3,
              }}
            />
            <div
              style={{
                position: "relative",
                backgroundColor: colors.codeBackground,
                padding: "24px 48px",
                borderRadius: 16,
                fontFamily: fonts.mono,
                fontSize: 36,
                border: `1px solid ${colors.backgroundAlt}`,
                boxShadow: "0 4px 24px rgba(0, 0, 0, 0.4)",
              }}
            >
              <span style={{ color: colors.textDim }}>$ </span>
              <TypewriterText
                text="pip install sklab"
                startFrame={40}
                framesPerChar={3}
                fontSize={36}
                fontFamily={fonts.mono}
                fontWeight={500}
                color={colors.successGreenBright}
                showCursor={true}
              />
            </div>
          </div>
        </div>

        {/* GitHub link */}
        <div
          style={{
            opacity: Math.max(0, linkOpacity),
            fontFamily: fonts.sans,
            fontSize: 26,
            display: "flex",
            alignItems: "center",
            gap: 12,
          }}
        >
          <svg
            width="28"
            height="28"
            viewBox="0 0 24 24"
            fill={colors.textMuted}
          >
            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
          </svg>
          <span style={{ color: colors.textMuted }}>
            github.com/
            <span style={{ color: colors.text, fontWeight: 600 }}>
              cfeenux/sklab
            </span>
          </span>
        </div>
      </div>
    </AbsoluteFill>
  );
};
