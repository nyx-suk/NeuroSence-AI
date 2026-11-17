/* Hey Neuro — browser-only voice assistant for SYNAPTIQ.
   Pipeline:
   Wake Word -> Microphone -> Speech-to-Text -> Intent -> Validation -> Action -> TTS
*/

(function () {
  "use strict";

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  class WakeWordDetector {
    constructor(words) {
      this.words = words.map(w => w.toLowerCase());
    }
    containsWakeWord(text) {
      const normalized = String(text || "").toLowerCase().replace(/[.,!?]/g, " ");
      return this.words.some(w => normalized.includes(w));
    }
    stripWakeWord(text) {
      let result = String(text || "");
      for (const word of this.words) {
        const re = new RegExp("\\b" + word.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&") + "\\b", "i");
        result = result.replace(re, " ");
      }
      return result.replace(/\s+/g, " ").trim();
    }
  }

  class SpeechEngine {
    constructor(lang, callbacks) {
      this.lang = lang || "en-IN";
      this.callbacks = callbacks || {};
      this.recognition = null;
      this.mode = "wake";
      this.active = false;
      this.processing = false;
      this.localTried = false;
    }

    async tryLocalRecognition() {
      if (this.localTried || !SpeechRecognition) return;
      this.localTried = true;
      try {
        if (typeof SpeechRecognition.available === "function") {
          const result = await SpeechRecognition.available({
            langs: [this.lang],
            sources: ["local"],
            processLocally: true
          });
          if (result === "available" || result === "downloadable") {
            if (typeof SpeechRecognition.install === "function" && result === "downloadable") {
              try { await SpeechRecognition.install({ langs: [this.lang] }); } catch (_) {}
            }
          }
        }
      } catch (_) {}
    }

    start(mode) {
      if (!SpeechRecognition || this.processing) return false;
      this.mode = mode || "wake";
      this.processing = true;

      const r = new SpeechRecognition();
      this.recognition = r;
      r.lang = this.lang;
      r.continuous = false;
      r.interimResults = true;
      r.maxAlternatives = 3;

      try {
        if (this.localTried) r.processLocally = true;
      } catch (_) {}

      let finalText = "";

      r.onstart = () => {
        this.active = true;
        this.callbacks.onStart?.(this.mode);
      };

      r.onresult = (event) => {
        let interim = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const txt = event.results[i][0].transcript;
          if (event.results[i].isFinal) finalText += txt + " ";
          else interim += txt;
        }
        this.callbacks.onText?.((finalText + interim).trim(), this.mode);
      };

      r.onerror = (event) => {
        this.callbacks.onError?.(event.error || "recognition-error");
      };

      r.onend = () => {
        this.processing = false;
        this.active = false;
        this.callbacks.onEnd?.(finalText.trim(), this.mode);
      };

      try {
        r.start();
        return true;
      } catch (e) {
        this.processing = false;
        this.active = false;
        this.callbacks.onError?.(e.message || "start-failed");
        return false;
      }
    }

    stop() {
      try { this.recognition?.stop(); } catch (_) {}
      this.processing = false;
      this.active = false;
    }
  }

  class IntentProcessor {
    constructor() {
      this.rules = [
        { id:"OPEN_GAMES", words:["go to games","open games","play games","brain games","cognitive games","show games"] },
        { id:"OPEN_HOME", words:["go home","open home","take me home","home page","show home"] },
        { id:"SHOW_LOCATION", words:["show location","where is the patient","where is patient","show patient location","find the patient","patient location"] },
        { id:"SHOW_ALERTS", words:["show alerts","show safety alerts","open alerts","safety alerts","show emergencies"] },
        { id:"SHOW_MEDICATION", words:["show medications","show medication","open medications","my reminders","show reminders","medications"] },
        { id:"SHOW_ACTIVITY", words:["show activity","show my activity","show progress","show cognitive performance","my activity"] },
        { id:"SHOW_SAFE_ZONES", words:["show safe zones","open safe zones","show geofence","show geofencing","safe zones"] },
        { id:"CALL_CAREGIVER", words:["call caregiver","call my caregiver","call emergency contact","call my emergency contact","phone caregiver"] },
        { id:"HELP", words:["help","what can you do","show commands","voice commands"] }
      ];
    }

    normalize(text) {
      return String(text || "").toLowerCase().replace(/[.,!?]/g," ").replace(/\s+/g," ").trim();
    }

    classify(text) {
      const q = this.normalize(text);
      for (const rule of this.rules) {
        if (rule.words.some(p => q.includes(p))) return rule.id;
      }
      return null;
    }
  }

  class SafeCommandValidator {
    constructor() {
      this.allowed = new Set([
        "OPEN_GAMES","OPEN_HOME","SHOW_LOCATION","SHOW_ALERTS",
        "SHOW_MEDICATION","SHOW_ACTIVITY","SHOW_SAFE_ZONES",
        "CALL_CAREGIVER","HELP"
      ]);
    }
    isAllowed(intent) {
      return this.allowed.has(intent);
    }
    isYes(text) {
      return /^(yes|yeah|yep|sure|okay|ok|confirm|do it|please do|call)$/i.test(String(text || "").trim());
    }
    isNo(text) {
      return /^(no|nope|cancel|stop|don't|do not)$/i.test(String(text || "").trim());
    }
  }

  class WebsiteActionHandlers {
    constructor(config) {
      this.config = config || {};
    }

    navigate(action) {
      const url = new URL(window.top.location.href);
      url.searchParams.set("neura_action", action);
      try {
        window.top.location.href = url.toString();
      } catch (_) {
        window.parent.location.href = url.toString();
      }
    }

    callCaregiver() {
      const phone = String(this.config.caregiverPhone || "");
      const safe = phone.replace(/[^0-9+*#;,]/g, "");
      if (!safe) return false;
      try {
        window.top.location.href = "tel:" + safe;
      } catch (_) {
        window.location.href = "tel:" + safe;
      }
      return true;
    }
  }

  class NeuraAssistant {
    constructor(root, config) {
      this.root = root;
      this.config = config || {};
      this.statusEl = root.querySelector("#neura-status");
      this.enableBtn = root.querySelector("#neura-enable");
      this.stopBtn = root.querySelector("#neura-stop");
      this.wake = new WakeWordDetector([
        "hey neuro", "hey neura", "hey nura", "hey neural", "hey nera", "hey newra"
      ]);
      this.intent = new IntentProcessor();
      this.validator = new SafeCommandValidator();
      this.actions = new WebsiteActionHandlers(this.config);
      this.engine = new SpeechEngine(this.config.lang || "en-IN", {
        onStart: (mode) => this.setStatus(mode === "wake"
          ? "👂 Listening for “Hey Neuro”..."
          : "🎤 Listening for your command..."),
        onText: (text, mode) => {
          if (mode === "wake") {
            if (this.wake.containsWakeWord(text)) {
              this.engine.stop();
              const remainder = this.wake.stripWakeWord(text);
              this.speak("Yes, how can I help?");
              setTimeout(() => {
                if (remainder) this.handleCommand(remainder);
                else this.engine.start("command");
              }, 500);
            }
          }
        },
        onError: (err) => {
          if (err === "not-allowed" || err === "service-not-allowed") {
            this.setStatus("Microphone permission is blocked. Allow microphone access for this site.");
          } else if (err === "network") {
            this.setStatus("Speech recognition needs a supported browser connection.");
          } else if (this.enabled) {
            this.setStatus("Voice paused. Press Enable Voice to start again.");
          }
        },
        onEnd: (text, mode) => {
          if (!this.enabled) return;
          if (mode === "wake") {
            setTimeout(() => {
              if (this.enabled) this.engine.start("wake");
            }, 250);
          } else if (mode === "command" && text) {
            this.handleCommand(text);
          }
        }
      });

      this.enabled = false;
      this.confirmingCall = false;

      this.enableBtn?.addEventListener("click", () => this.enable());
      this.stopBtn?.addEventListener("click", () => this.disable());

      if (!SpeechRecognition) {
        this.setStatus("Your browser does not support SpeechRecognition. Use current Chrome or Edge.");
        if (this.enableBtn) this.enableBtn.disabled = true;
      }
    }

    setStatus(text) {
      if (this.statusEl) this.statusEl.textContent = text;
      this.root.classList.toggle("neura-listening", /Listening/.test(text));
    }

    speak(text) {
      if (!("speechSynthesis" in window)) return;
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = this.config.lang || "en-IN";
      u.rate = 0.88;
      u.pitch = 1.0;
      const voices = window.speechSynthesis.getVoices();
      const exact = voices.find(v => v.lang.toLowerCase() === u.lang.toLowerCase());
      const prefix = voices.find(v => v.lang.toLowerCase().startsWith((u.lang || "en").split("-")[0]));
      if (exact || prefix) u.voice = exact || prefix;
      window.speechSynthesis.speak(u);
    }

    async enable() {
      if (!SpeechRecognition) return;
      try {
        if (navigator.mediaDevices?.getUserMedia) {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          stream.getTracks().forEach(t => t.stop());
        }
      } catch (_) {
        this.setStatus("Please allow microphone access, then press Enable Voice again.");
        return;
      }

      this.enabled = true;
      this.enableBtn?.classList.add("neura-hidden");
      this.stopBtn?.classList.remove("neura-hidden");
      this.setStatus("👂 Listening for “Hey Neuro”...");
      await this.engine.tryLocalRecognition();
      this.engine.start("wake");
      this.speak("Hey Neuro is ready.");
    }

    disable() {
      this.enabled = false;
      this.confirmingCall = false;
      this.engine.stop();
      window.speechSynthesis?.cancel();
      this.stopBtn?.classList.add("neura-hidden");
      this.enableBtn?.classList.remove("neura-hidden");
      this.setStatus("Voice paused. Press Enable Voice to start again.");
    }

    handleCommand(text) {
      if (this.confirmingCall) {
        if (this.validator.isYes(text)) {
          this.confirmingCall = false;
          if (this.actions.callCaregiver()) {
            this.speak("Calling your emergency contact.");
            this.setStatus("Calling your emergency contact...");
          } else {
            this.speak("I could not find a saved emergency contact number.");
            this.setStatus("No callable caregiver number is saved.");
            this.resumeWake();
          }
          return;
        }
        if (this.validator.isNo(text)) {
          this.confirmingCall = false;
          this.speak("Okay, I will not call.");
          this.resumeWake();
          return;
        }
        this.speak("Please say yes to call, or no to cancel.");
        setTimeout(() => this.engine.start("command"), 700);
        return;
      }

      const intent = this.intent.classify(text);
      if (!intent || !this.validator.isAllowed(intent)) {
        this.speak("Sorry, I didn't understand that command.");
        this.setStatus("Command not recognized. Say “Hey Neuro” and try again.");
        this.resumeWake();
        return;
      }

      const messages = {
        OPEN_GAMES:"Opening games.",
        OPEN_HOME:"Going home.",
        SHOW_LOCATION:"Showing the patient's location.",
        SHOW_ALERTS:"Opening safety alerts.",
        SHOW_MEDICATION:"Opening medications and reminders.",
        SHOW_ACTIVITY:"Showing activity.",
        SHOW_SAFE_ZONES:"Showing safe zones.",
        HELP:"You can say go to games, go home, show location, show alerts, show medications, show activity, show safe zones, or call caregiver."
      };

      if (intent === "CALL_CAREGIVER") {
        this.confirmingCall = true;
        this.speak("Do you want me to call your emergency contact?");
        this.setStatus("Please say yes to call, or no to cancel.");
        setTimeout(() => this.engine.start("command"), 900);
        return;
      }

      this.speak(messages[intent] || "Okay.");
      this.setStatus(messages[intent] || "Okay.");
      setTimeout(() => this.actions.navigate(intent), 650);
    }

    resumeWake() {
      if (!this.enabled) return;
      setTimeout(() => this.engine.start("wake"), 700);
    }
  }

  window.NeuraAssistant = NeuraAssistant;
  window.WakeWordDetector = WakeWordDetector;
  window.IntentProcessor = IntentProcessor;
  window.SafeCommandValidator = SafeCommandValidator;
  window.WebsiteActionHandlers = WebsiteActionHandlers;
})();
