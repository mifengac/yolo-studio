/* 全局壳层：路由、API、状态 */
const YS = {
  api: async function (url, opts = {}) {
    const res = await fetch(url, opts);
    let data = null;
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      data = await res.json();
    } else {
      data = await res.text();
    }
    if (!res.ok) {
      const msg = (data && data.detail) || (typeof data === "string" ? data : JSON.stringify(data));
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return data;
  },
  statusCn(s) {
    return ({
      pending: "排队中",
      running: "运行中",
      success: "成功",
      failed: "失败",
      canceled: "已取消",
      interrupted: "已中断",
    })[s] || s;
  },
  fmtSec(sec) {
    sec = Math.max(0, Math.round(sec || 0));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    if (h) return `${h}小时${m}分`;
    return `${m}分${sec % 60}秒`;
  },
};

window.YSApp = null;

document.addEventListener("DOMContentLoaded", () => {
  const { createApp } = Vue;
  window.YSApp = createApp({
    data() {
      return {
        page: "datasets",
        datasets: [],
        currentDs: null,
        showCreate: false,
        newDs: { name: "", classes: "wheelie, multi rider", notes: "" },
        toast: "",
        taskMsg: {},
        sysInfo: null,
        trainForm: {
          dataset_id: "",
          group: "finetune",
          base_model: "0517_yolo26n_wheelie_multi-rider.pt",
          epochs: 20,
          imgsz: 416,
          batch: 16,
          freeze: 10,
          only_confirmed: true,
          force_long: false,
        },
        baseModels: { scratch: [], finetune: [] },
        estimate: null,
        trainJobs: [],
        activeJob: null,
        trainLog: "",
        modelList: [],
        _logEs: null,
        _chart: null,
        _poll: null,
      };
    },
    computed: {
      baseModelOptions() {
        return this.trainForm.group === "scratch"
          ? this.baseModels.scratch || []
          : this.baseModels.finetune || [];
      },
    },
    methods: {
      statusCn: YS.statusCn,
      fmtSec: YS.fmtSec,
      showToast(msg) {
        this.toast = msg;
        setTimeout(() => { if (this.toast === msg) this.toast = ""; }, 3200);
      },
      go(page) {
        this.page = page;
        if (page === "datasets") this.loadDatasets();
        if (page === "train") {
          this.loadDatasets();
          this.loadBaseModels();
          this.loadJobs();
          if (this.currentDs) this.trainForm.dataset_id = this.currentDs.id;
        }
        if (page === "models") this.loadModels();
        if (page === "labeler" && this.currentDs) {
          this.$nextTick(() => {
            if (window.YSLabeler) window.YSLabeler.mount(this.currentDs, this);
          });
        }
      },
      async loadDatasets() {
        this.datasets = await YS.api("/api/datasets");
      },
      async createDataset() {
        try {
          const classes = this.newDs.classes.split(/[,，;；]/).map((s) => s.trim()).filter(Boolean);
          const d = await YS.api("/api/datasets", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: this.newDs.name, classes, notes: this.newDs.notes }),
          });
          this.showCreate = false;
          this.newDs = { name: "", classes: "wheelie, multi rider", notes: "" };
          await this.loadDatasets();
          this.showToast("数据集已创建");
          this.openLabeler(d);
        } catch (e) {
          this.showToast(e.message);
        }
      },
      openLabeler(d) {
        this.currentDs = d;
        this.trainForm.dataset_id = d.id;
        this.go("labeler");
      },
      async removeDataset(d) {
        if (!confirm(`确定删除数据集「${d.name}」？图片与标注会一并删除。`)) return;
        await YS.api(`/api/datasets/${d.id}`, { method: "DELETE" });
        await this.loadDatasets();
        if (this.currentDs && this.currentDs.id === d.id) this.currentDs = null;
        this.showToast("已删除");
      },
      async uploadFiles(d) {
        const input = document.createElement("input");
        input.type = "file";
        input.multiple = true;
        input.accept = "image/*";
        input.onchange = async () => {
          const fd = new FormData();
          for (const f of input.files) fd.append("files", f);
          try {
            const r = await YS.api(`/api/datasets/${d.id}/import/files`, { method: "POST", body: fd });
            this.showToast(`导入 ${r.imported} 张（跳过重复 ${r.skipped_dup}）`);
            await this.loadDatasets();
          } catch (e) {
            this.showToast(e.message);
          }
        };
        input.click();
      },
      async uploadZip(d) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".zip";
        input.onchange = async () => {
          const fd = new FormData();
          fd.append("file", input.files[0]);
          try {
            const r = await YS.api(`/api/datasets/${d.id}/import/zip`, { method: "POST", body: fd });
            this.showToast(`ZIP 导入 ${r.imported} 张，含标注 ${r.labeled}`);
            await this.loadDatasets();
          } catch (e) {
            this.showToast(e.message);
          }
        };
        input.click();
      },
      async uploadVideo(d) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "video/*";
        input.onchange = async () => {
          const fd = new FormData();
          fd.append("file", input.files[0]);
          try {
            this.taskMsg[d.id] = "正在抽帧…";
            const r = await YS.api(`/api/datasets/${d.id}/import/video`, { method: "POST", body: fd });
            this.taskMsg[d.id] = `抽帧完成 ${r.frames} 帧`;
            await this.loadDatasets();
          } catch (e) {
            this.taskMsg[d.id] = e.message;
          }
        };
        input.click();
      },
      async pollTask(taskId, dsId) {
        for (let i = 0; i < 3600; i++) {
          const t = await YS.api(`/api/tasks/${taskId}`);
          this.taskMsg[dsId] = `${t.message || t.status} (${Math.round(t.progress || 0)}%)`;
          if (["success", "failed", "canceled"].includes(t.status)) {
            if (t.error) this.taskMsg[dsId] += " · " + t.error;
            await this.loadDatasets();
            return t;
          }
          await new Promise((r) => setTimeout(r, 1500));
        }
      },
      async runAutolabel(d) {
        try {
          const t = await YS.api(`/api/datasets/${d.id}/autolabel`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ scope: "unlabeled", conf: 0.25, overwrite: false }),
          });
          this.pollTask(t.id, d.id);
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async runOpenvocab(d) {
        const prompts = prompt(
          "开放词表提示词（逗号分隔）",
          "motorcycle with front wheel lifted, three people on one motorcycle"
        );
        if (!prompts) return;
        try {
          const t = await YS.api(`/api/datasets/${d.id}/autolabel/openvocab`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              prompts: prompts.split(/[,，]/).map((s) => s.trim()).filter(Boolean),
              conf: 0.1,
              preview_limit: 20,
            }),
          });
          this.pollTask(t.id, d.id);
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async runDedup(d) {
        try {
          const t = await YS.api(`/api/datasets/${d.id}/dedup`, { method: "POST" });
          this.pollTask(t.id, d.id);
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async loadBaseModels() {
        this.baseModels = await YS.api("/api/models/base-options");
        this.onGroupChange();
      },
      onGroupChange() {
        const opts = this.baseModelOptions.filter((m) => m.available);
        if (opts.length) {
          if (!opts.find((m) => m.id === this.trainForm.base_model)) {
            this.trainForm.base_model = opts[0].id;
          }
        }
        if (this.trainForm.group === "finetune") this.trainForm.epochs = 20;
        else this.trainForm.epochs = 40;
        this.refreshEstimate();
      },
      async refreshEstimate() {
        if (!this.trainForm.dataset_id) return;
        try {
          this.estimate = await YS.api("/api/train/estimate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              dataset_id: this.trainForm.dataset_id,
              base_model: this.trainForm.base_model,
              epochs: this.trainForm.epochs,
              imgsz: this.trainForm.imgsz,
              batch: this.trainForm.batch,
              freeze: this.trainForm.freeze,
              only_confirmed: this.trainForm.only_confirmed,
            }),
          });
        } catch (e) {
          this.estimate = { human: "—", tips: [e.message], seconds: 0, cache: "-", seconds_per_epoch: 0 };
        }
      },
      async startTrain() {
        try {
          const j = await YS.api("/api/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(this.trainForm),
          });
          this.showToast("训练已提交 " + j.id);
          await this.loadJobs();
          this.watchJob(j);
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async loadJobs() {
        this.trainJobs = await YS.api("/api/train");
      },
      async cancelTrain(j) {
        await YS.api(`/api/train/${j.id}/cancel`, { method: "POST" });
        await this.loadJobs();
      },
      async resumeTrain(j) {
        try {
          await YS.api(`/api/train/${j.id}/resume`, { method: "POST" });
          this.showToast("已继续训练");
          await this.loadJobs();
          this.watchJob(j);
        } catch (e) {
          this.showToast(e.message);
        }
      },
      async publishTrain(j) {
        try {
          const m = await YS.api(`/api/train/${j.id}/publish`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}",
          });
          this.showToast("已发布 " + m.name);
          this.loadModels();
        } catch (e) {
          this.showToast(e.message);
        }
      },
      watchJob(j) {
        this.activeJob = j;
        this.trainLog = "";
        if (this._logEs) this._logEs.close();
        this._logEs = new EventSource(`/api/train/${j.id}/logs`);
        this._logEs.onmessage = (ev) => {
          this.trainLog += ev.data + "\n";
          this.$nextTick(() => {
            const el = this.$refs.logbox;
            if (el) el.scrollTop = el.scrollHeight;
          });
        };
        this._pollMetrics(j.id);
      },
      async _pollMetrics(jobId) {
        if (this._poll) clearInterval(this._poll);
        const tick = async () => {
          try {
            const r = await YS.api(`/api/train/${jobId}/metrics`);
            this._drawChart(r.series || []);
            await this.loadJobs();
            const j = this.trainJobs.find((x) => x.id === jobId);
            if (j) this.activeJob = j;
            if (j && !["running", "pending"].includes(j.status)) {
              clearInterval(this._poll);
              this._poll = null;
            }
          } catch (_) {}
        };
        tick();
        this._poll = setInterval(tick, 5000);
      },
      _drawChart(series) {
        const el = document.getElementById("train-chart");
        if (!el || typeof echarts === "undefined") return;
        if (!this._chart) this._chart = echarts.init(el);
        const epochs = series.map((s) => s.epoch);
        this._chart.setOption({
          backgroundColor: "transparent",
          tooltip: { trigger: "axis" },
          legend: { data: ["box_loss", "cls_loss", "mAP50"], textStyle: { color: "#94a3b8" } },
          xAxis: { type: "category", data: epochs, axisLabel: { color: "#94a3b8" } },
          yAxis: [
            { type: "value", name: "loss", axisLabel: { color: "#94a3b8" }, splitLine: { lineStyle: { color: "#334155" } } },
            { type: "value", name: "mAP", max: 1, axisLabel: { color: "#94a3b8" }, splitLine: { show: false } },
          ],
          series: [
            { name: "box_loss", type: "line", data: series.map((s) => s.box_loss), smooth: true },
            { name: "cls_loss", type: "line", data: series.map((s) => s.cls_loss), smooth: true },
            { name: "mAP50", type: "line", yAxisIndex: 1, data: series.map((s) => s.mAP50), smooth: true },
          ],
        });
      },
      async loadModels() {
        this.modelList = await YS.api("/api/models");
      },
      async setDefaultModel(m) {
        await YS.api(`/api/models/${m.id}/set-default-autolabel`, { method: "POST" });
        this.showToast("已设为默认预标注模型");
        this.loadModels();
      },
      async delModel(m) {
        if (!confirm("确定删除模型 " + m.name + "？")) return;
        await YS.api(`/api/models/${m.id}`, { method: "DELETE" });
        this.loadModels();
      },
    },
    async mounted() {
      try {
        this.sysInfo = await YS.api("/api/system/info");
      } catch (_) {}
      await this.loadDatasets();
    },
  }).mount("#app");
});
