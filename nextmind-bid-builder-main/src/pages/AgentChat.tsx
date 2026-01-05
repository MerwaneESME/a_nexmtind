import { useMemo, useRef, useState } from "react";
import { Navigation } from "@/components/Navigation";
import { Button } from "@/components/ui/button";
import { Loader2, Paperclip, Send } from "lucide-react";

type Msg = { role: "user" | "assistant"; content: string };

const API_BASE = (import.meta as any).env?.VITE_AGENT_API ?? "http://127.0.0.1:8000";

const formatPayload = (payload: any) => {
  if (!payload) return "Réponse vide";
  if (payload.reply) return payload.reply;

  const parts: string[] = [];
  if (payload.error) parts.push(`Erreur: ${payload.error} ${payload.details ?? ""}`.trim());
  if (payload.totals) parts.push(`Totaux:\n${JSON.stringify(payload.totals, null, 2)}`);
  if (payload.inconsistencies?.length)
    parts.push(`Incohérences:\n${JSON.stringify(payload.inconsistencies, null, 2)}`);
  if (payload.data) parts.push(`Données:\n${JSON.stringify(payload.data, null, 2)}`);
  if (payload.supabase) parts.push(`Supabase:\n${JSON.stringify(payload.supabase, null, 2)}`);
  return parts.join("\n\n") || "OK";
};

const AgentChat = () => {
  const [messages, setMessages] = useState<Msg[]>([
    {
      role: "assistant",
      content:
        "Bonjour ! Posez vos questions ou envoyez un devis/facture (PDF/DOCX/PNG/JPG). Je détecte le type, j'extrais les données et je peux générer des devis en JSON pour Supabase.",
    },
  ]);
  const [input, setInput] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    requestAnimationFrame(() => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    });
  };

  const pushMessage = (msg: Msg) => {
    setMessages((prev) => [...prev, msg]);
    scrollToBottom();
  };

  const handleSend = async () => {
    if (loading) return;
    if (!file && !input.trim()) return;

    if (file) {
      pushMessage({ role: "user", content: `Fichier: ${file.name}` });
      setLoading(true);
      try {
        const fd = new FormData();
        fd.append("file", file);
        const resp = await fetch(`${API_BASE}/analyze`, { method: "POST", body: fd });
        const text = await resp.text();
        let json: any;
        try {
          json = JSON.parse(text);
        } catch {
          json = { error: "parse_error", details: text };
        }
        pushMessage({ role: "assistant", content: formatPayload(json) });
      } catch (err: any) {
        pushMessage({ role: "assistant", content: `Erreur: ${err.message}` });
      } finally {
        setFile(null);
        setLoading(false);
      }
      return;
    }

    const text = input.trim();
    setInput("");
    pushMessage({ role: "user", content: text });
    setLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const json = await resp.json();
      pushMessage({ role: "assistant", content: formatPayload(json) });
    } catch (err: any) {
      pushMessage({ role: "assistant", content: `Erreur: ${err.message}` });
    } finally {
      setLoading(false);
    }
  };

  const fileLabel = useMemo(() => (file ? file.name : "Joindre un fichier"), [file]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#f5f8ff] to-white text-slate-900">
      <Navigation />
      <main className="container py-10 space-y-6">
        <div className="space-y-2">
          <p className="text-xs font-bold text-sky-600 uppercase tracking-wide">Assistant BTP</p>
          <h1 className="text-3xl md:text-4xl font-extrabold text-slate-900">
            Agent IA Nextmind
          </h1>
          <p className="text-slate-600">
            Discute avec l&apos;agent, pose tes questions BTP ou dépose un devis/facture. L&apos;IA
            détecte automatiquement le type de document et renvoie un JSON structuré prêt pour
            Supabase.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white shadow-xl shadow-slate-200/50">
          <div
            ref={scrollRef}
            className="max-h-[70vh] overflow-y-auto px-6 pt-6 pb-4 space-y-3 bg-slate-50/60 border-b border-slate-100"
          >
            {messages.map((m, idx) => (
              <div
                key={idx}
                className={`rounded-xl px-4 py-3 max-w-4xl whitespace-pre-wrap leading-relaxed ${
                  m.role === "user"
                    ? "self-end bg-sky-100 text-slate-900 border border-sky-200"
                    : "bg-white border border-slate-200 text-slate-900 shadow-sm"
                }`}
              >
                {m.content}
              </div>
            ))}
            {loading && (
              <div className="flex items-center gap-2 text-slate-500 text-sm">
                <Loader2 className="h-4 w-4 animate-spin" /> L&apos;agent réfléchit...
              </div>
            )}
          </div>

          <div className="flex flex-col gap-3 p-4">
            <div className="flex flex-wrap gap-2 items-center">
              <label className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-dashed border-slate-300 cursor-pointer bg-white hover:bg-slate-50 text-sm font-medium text-slate-700">
                <Paperclip className="h-4 w-4" />
                {fileLabel}
                <input
                  type="file"
                  accept=".pdf,.docx,.png,.jpg,.jpeg"
                  className="hidden"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
              </label>
              {file && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-red-500"
                  onClick={() => setFile(null)}
                >
                  Retirer le fichier
                </Button>
              )}
            </div>

            <div className="flex items-end gap-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Message ou instructions..."
                className="flex-1 min-h-[56px] max-h-40 rounded-xl border border-slate-300 bg-white px-4 py-3 text-base text-slate-900 shadow-inner focus:outline-none focus:ring-2 focus:ring-sky-300 focus:border-sky-300"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                disabled={loading}
              />
              <Button
                className="h-[56px] px-5 bg-gradient-to-r from-sky-500 to-sky-600 text-white font-semibold shadow-lg hover:shadow-xl"
                onClick={handleSend}
                disabled={loading}
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                <span className="ml-2">Envoyer</span>
              </Button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default AgentChat;
