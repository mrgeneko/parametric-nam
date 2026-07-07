using Circuit;
using ComputerAlgebra;
using Util;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace livespice_cli
{
    class Program
    {
        static void Main(string[] args)
        {
            string inputPath = null, outputPath = null, circuitPath = null, paramsStr = null, speakerName = null;
            string netlistPath = null;
            int sampleRate = 48000, oversample = 2, iterations = 8;

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--input": inputPath = args[++i]; break;
                    case "--output": outputPath = args[++i]; break;
                    case "--circuit": circuitPath = args[++i]; break;
                    case "--params": paramsStr = args[++i]; break;
                    case "--speaker": speakerName = args[++i]; break;
                    case "--sr": sampleRate = int.Parse(args[++i]); break;
                    case "--oversample": oversample = int.Parse(args[++i]); break;
                    case "--iterations": iterations = int.Parse(args[++i]); break;
                    // Dump the authoritative netlist (components + node connectivity) as JSON
                    // and exit — the foundation for the ngspice-backend .schx translator.
                    case "--netlist": netlistPath = args[++i]; break;
                }
            }

            // --netlist only needs --circuit; the audio flags aren't required.
            if (circuitPath == null || (netlistPath == null && (inputPath == null || outputPath == null)))
            {
                Console.Error.WriteLine("Usage: livespice_cli --input in.wav --output out.wav --circuit circuit.schx [--params k=v,...] [--speaker S1] [--sr 48000] [--oversample 2] [--iterations 8]");
                Console.Error.WriteLine("   or: livespice_cli --circuit circuit.schx --netlist out.json   (dump netlist and exit)");
                Environment.Exit(1);
            }

            var log = new ConsoleLog { Verbosity = MessageType.Warning };

            // load circuit
            var schematic = Schematic.Load(circuitPath, log);
            var circuit = schematic.Build(log);

            if (netlistPath != null)
            {
                DumpNetlist(circuit, netlistPath);
                return;
            }

            // set knob parameters
            if (paramsStr != null)
            {
                foreach (var kv in paramsStr.Split(','))
                {
                    var parts = kv.Split('=');
                    if (parts.Length != 2) continue;
                    var name = parts[0].Trim();
                    var value = double.Parse(parts[1].Trim());

                    // Set ALL pots whose name matches, not just the first. This lets a
                    // ganged (multi-section) pot be modeled as several Potentiometer
                    // components sharing one Name — e.g. the Klon Centaur's dual-gang
                    // Gain (a 100k + a 1k section) both track a single "Gain" knob value.
                    var pots = circuit.Components
                        .OfType<IPotControl>()
                        .Cast<Component>()
                        .Where(c => c.Name.Trim() == name)
                        .ToList();

                    if (pots.Count > 0)
                    {
                        foreach (var p in pots)
                            ((IPotControl)p).PotValue = value;
                        continue;
                    }

                    // Same for switches: set every matching section (ganged switches).
                    var switches = circuit.Components
                        .OfType<SinglePoleSwitch>()
                        .Where(c => c.Name.Trim() == name)
                        .ToList();

                    foreach (var sw in switches)
                        sw.Position = (int)value;
                }
            }

            // find input component
            var inputExpr = circuit.Components
                .OfType<Input>()
                .Select(i => i.In)
                .DefaultIfEmpty("V[t]")
                .Single();

            // find speaker output(s)
            var allSpeakers = circuit.Components.OfType<Speaker>().ToList();
            List<Speaker> speakers;
            if (speakerName != null)
            {
                speakers = allSpeakers.Where(s => s.Name == speakerName).ToList();
                if (speakers.Count == 0)
                {
                    var names = string.Join(", ", allSpeakers.Select(s => s.Name));
                    Console.Error.WriteLine($"Speaker '{speakerName}' not found. Available: {names}");
                    Environment.Exit(1);
                }
            }
            else
            {
                speakers = allSpeakers;
            }

            Expression outputExpr;
            if (speakers.Count == 0)
            {
                // fallback: use first node voltage
                outputExpr = circuit.Nodes.Select(n => n.V).FirstOrDefault() ?? 0;
            }
            else
            {
                var sum = (Expression)0;
                foreach (var s in speakers)
                    sum += s.Out;
                outputExpr = sum;
            }

            // read input WAV
            var (inSamples, inSampleRate) = ReadWav(inputPath);
            int N = inSamples.Length;
            int outSampleRate = sampleRate > 0 ? sampleRate : inSampleRate;

            // build simulation
            var ts = TransientSolution.Solve(
                circuit.Analyze(),
                (Real)1 / (outSampleRate * oversample),
                log);

            var sim = new Simulation(ts)
            {
                Oversample = oversample,
                Iterations = iterations,
                Input = new[] { inputExpr },
                Output = new[] { outputExpr },
            };

            // process in chunks
            double[] outputBuffer = new double[N];
            int chunk = 4096;
            int offset = 0;
            while (offset < N)
            {
                int n = Math.Min(chunk, N - offset);
                double[] chunkIn = new double[n];
                Array.Copy(inSamples, offset, chunkIn, 0, n);
                double[] chunkOut = new double[n];
                sim.Run(chunkIn, chunkOut);
                Array.Copy(chunkOut, 0, outputBuffer, offset, n);
                offset += n;
            }

            // write output WAV
            WriteWavFloat(outputPath, outputBuffer, outSampleRate);

            double dur = (double)N / outSampleRate;
            Console.Error.WriteLine($"Input:    {inputPath}");
            Console.Error.WriteLine($"Format:   {outSampleRate} Hz, 1 ch, 32-bit float  →  {dur:F3} sec mono");
            Console.Error.WriteLine($"Circuit:  {circuit.Name}  (oversample {oversample}x)");
            Console.Error.WriteLine($"Output:   {outputPath}  ({N} samples, 32-bit float mono)");
        }

        static (double[] samples, int sampleRate) ReadWav(string path)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var br = new BinaryReader(fs);

            // RIFF header
            var riff = Encoding.ASCII.GetString(br.ReadBytes(4));
            if (riff != "RIFF") throw new Exception("Not a RIFF file");
            br.ReadInt32(); // file size
            var wave = Encoding.ASCII.GetString(br.ReadBytes(4));
            if (wave != "WAVE") throw new Exception("Not a WAVE file");

            int channels = 0, sampleRate = 0, bitsPerSample = 0;
            List<byte> data = null;

            while (fs.Position < fs.Length)
            {
                var chunkId = Encoding.ASCII.GetString(br.ReadBytes(4));
                int chunkSize = br.ReadInt32();

                if (chunkId == "fmt ")
                {
                    var audioFormat = br.ReadInt16(); // 1=PCM, 3=IEEE float
                    channels = br.ReadInt16();
                    sampleRate = br.ReadInt32();
                    br.ReadInt32(); // byte rate
                    br.ReadInt16(); // block align
                    bitsPerSample = br.ReadInt16();
                    if (chunkSize > 16)
                        br.ReadBytes(chunkSize - 16);
                }
                else if (chunkId == "data")
                {
                    data = new List<byte>(br.ReadBytes(chunkSize));
                }
                else
                {
                    br.ReadBytes(chunkSize);
                }
            }

            if (data == null) throw new Exception("No data chunk found");

            // Snapshot the data chunk to an array ONCE. Previously the 16- and 32-bit
            // branches called data.ToArray() per sample, copying the whole ~MB chunk on
            // every iteration — O(N^2), which made large 16/32-bit-float WAVs appear to
            // hang inside ReadWav (the 24-bit branch indexed directly and was unaffected).
            byte[] buf = data.ToArray();

            int bytesPerSample = bitsPerSample / 8;
            int totalSamples = buf.Length / bytesPerSample / channels;
            double[] samples = new double[totalSamples];

            for (int i = 0; i < totalSamples; i++)
            {
                double sample = 0;
                int byteOffset = i * channels * bytesPerSample;

                if (bitsPerSample == 16)
                {
                    short val = BitConverter.ToInt16(buf, byteOffset);
                    sample = val / 32768.0;
                }
                else if (bitsPerSample == 24)
                {
                    int val = buf[byteOffset] | (buf[byteOffset + 1] << 8) | (buf[byteOffset + 2] << 16);
                    if ((val & 0x800000) != 0) val |= unchecked((int)0xFF000000);
                    sample = val / 8388608.0;
                }
                else if (bitsPerSample == 32)
                {
                    float val = BitConverter.ToSingle(buf, byteOffset);
                    sample = val;
                }
                else
                {
                    throw new Exception($"Unsupported bits per sample: {bitsPerSample}");
                }

                samples[i] = sample;
            }

            return (samples, sampleRate);
        }

        static void WriteWavFloat(string path, double[] samples, int sampleRate)
        {
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var bw = new BinaryWriter(fs);

            int dataSize = samples.Length * 4;
            int fileSize = 36 + dataSize;

            // RIFF header
            bw.Write(Encoding.ASCII.GetBytes("RIFF"));
            bw.Write(fileSize);
            bw.Write(Encoding.ASCII.GetBytes("WAVE"));

            // fmt chunk
            bw.Write(Encoding.ASCII.GetBytes("fmt "));
            bw.Write(16); // chunk size
            bw.Write((short)3); // IEEE float
            bw.Write((short)1); // mono
            bw.Write(sampleRate);
            bw.Write(sampleRate * 4); // byte rate
            bw.Write((short)4); // block align
            bw.Write((short)32); // bits per sample

            // data chunk
            bw.Write(Encoding.ASCII.GetBytes("data"));
            bw.Write(dataSize);
            foreach (var s in samples)
                bw.Write((float)s);
        }

        // -------------------------------------------------------------------
        // Netlist dump — authoritative components + node connectivity as JSON.
        // Consumed by the ngspice-backend .schx translator (batch_harness).
        // -------------------------------------------------------------------
        static string JsonStr(string s)
        {
            if (s == null) return "null";
            var sb = new StringBuilder("\"");
            foreach (char c in s)
            {
                if (c == '"' || c == '\\') sb.Append('\\').Append(c);
                else if (c == '\n') sb.Append("\\n");
                else if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4"));
                else sb.Append(c);
            }
            return sb.Append('"').ToString();
        }

        static string CompValue(Component c)
        {
            switch (c)
            {
                case Resistor r: return r.Resistance.ToString();
                case Capacitor cap: return cap.Capacitance.ToString();
                case Potentiometer p: return p.Resistance.ToString();
                case Rail rail: return rail.Voltage.ToString();
                case CenterTapTransformer tx: return tx.Turns.ToString();
                default: return null;
            }
        }

        static void DumpNetlist(Circuit.Circuit circuit, string path)
        {
            var sb = new StringBuilder();
            sb.Append("{\n  \"components\": [\n");
            bool first = true;
            foreach (var c in circuit.Components)
            {
                if (!first) sb.Append(",\n");
                first = false;
                sb.Append("    {\"name\": ").Append(JsonStr(c.Name));
                sb.Append(", \"type\": ").Append(JsonStr(c.GetType().Name));
                sb.Append(", \"value\": ").Append(JsonStr(CompValue(c)));
                sb.Append(", \"isPot\": ").Append(c is IPotControl ? "true" : "false");
                // All scalar properties (tube model params, Model enum, R/C values, ...)
                // so the ngspice translator can rebuild the exact device.
                sb.Append(", \"params\": {");
                bool pf = true;
                foreach (var prop in c.GetType().GetProperties())
                {
                    if (!prop.CanRead || prop.GetIndexParameters().Length > 0) continue;
                    object val;
                    try { val = prop.GetValue(c); } catch { continue; }
                    if (val == null) continue;
                    var vt = val.GetType();
                    if (val is double || val is int || vt.IsEnum || vt.Name == "Quantity")
                    {
                        if (!pf) sb.Append(", ");
                        pf = false;
                        sb.Append(JsonStr(prop.Name)).Append(": ").Append(JsonStr(val.ToString()));
                    }
                }
                sb.Append("}");
                sb.Append(", \"terminals\": [");
                bool tf = true;
                foreach (var t in c.Terminals)
                {
                    if (!tf) sb.Append(", ");
                    tf = false;
                    sb.Append("{\"name\": ").Append(JsonStr(t.Name));
                    sb.Append(", \"node\": ").Append(JsonStr(t.ConnectedTo != null ? t.ConnectedTo.Name : null));
                    sb.Append("}");
                }
                sb.Append("]}");
            }
            sb.Append("\n  ]\n}\n");
            File.WriteAllText(path, sb.ToString());
            Console.Error.WriteLine($"Wrote netlist: {circuit.Components.Count()} components -> {path}");
        }
    }
}
