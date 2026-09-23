import { Badge } from "../ui/Badge";

export function WelcomeHero() {
  return (
    <header className="mb-6 border-b-4 border-ink pb-6">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Badge tone="accent">Project LiPAD</Badge>
        <Badge>Hotspot · Pi 4B</Badge>
      </div>
      <h2 className="font-serif text-3xl leading-tight text-ink sm:text-4xl">
        Structural health analytics dashboard
      </h2>
      <p className="drop-cap mt-4 max-w-3xl font-serif text-base leading-relaxed text-ink">
        Project LiPAD (Light-based Inspection and Precision Analytics Drone) automates crack and
        corrosion monitoring with computer vision. Live telemetry from your Raspberry Pi streams
        over the local hotspot while inspection results populate the analysis overview.
      </p>
    </header>
  );
}
