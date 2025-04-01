<template>

    <div class="p-6 max-w-3xl mx-auto bg-white shadow rounded-md space-y-6">

<h1 class="text-xl font-bold text-gray-800">📝 Саммаризация большого текста</h1>

<div>
  <label class="block mb-1 text-sm font-medium text-gray-700">Введите текст</label>
  <textarea v-model="text" placeholder="Введите текст" rows="8" class="w-full border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"></textarea>
</div>

<div class="grid grid-cols-2 gap-4">
  <div>
    <label class="block text-sm font-medium text-gray-700">Размер чанка (символов)</label>
    <input type="number" v-model.number="params.chunk_size" class="w-full border rounded px-2 py-1" />
  </div>
  <div>
    <label class="block text-sm font-medium text-gray-700">Overlap (символов)</label>
    <input type="number" v-model.number="params.overlap" class="w-full border rounded px-2 py-1" />
  </div>
  <div>
    <label class="block text-sm font-medium text-gray-700">Temperature (чанки)</label>
    <input type="number" step="0.1" v-model.number="params.temp_chunk" class="w-full border rounded px-2 py-1" />
  </div>
  <div>
    <label class="block text-sm font-medium text-gray-700">Temperature (финал)</label>
    <input type="number" step="0.1" v-model.number="params.temp_final" class="w-full border rounded px-2 py-1" />
  </div>
</div>

<div>
  <label class="block mb-1 text-sm font-medium text-gray-700">Prompt для чанков (опционально)</label>
  <textarea v-model="params.chunk_prompt" rows="2" class="w-full border rounded px-2 py-1"></textarea>
</div>

<div>
  <label class="block mb-1 text-sm font-medium text-gray-700">Prompt для финального саммари (опционально)</label>
  <textarea v-model="params.final_prompt" rows="2" class="w-full border rounded px-2 py-1"></textarea>
</div>

<button @click="submitText"
  :disabled="loading"
  class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition">
  Сделать саммари
</button>

    <div v-if="loading">
      <p>⏳ Генерация саммари...</p>
    </div>

    <div v-if="totalChunks > 0" class="mb-2 text-sm text-gray-600">
      Обработка чанков: {{ processedChunks }} / {{ totalChunks }}
    </div>

    <div v-if="tokenCount > 0" class="mb-4 text-sm text-gray-500">
      Примерный размер текста:  {{ wordCount }} слов или {{ tokenCount }} токенов
    </div>

    <div v-if="chunkSummaries.length">
      <h2 class="font-bold mt-4">📦 Обработанные чанки:</h2>
      <ul>
        <li v-for="(chunk, index) in chunkSummaries" :key="index" class="mb-2">
          <strong>Chunk {{ chunk.chunk }}</strong> ({{ chunk.duration }} сек):
          <div class="text-gray-700">{{ chunk.summary }}</div>
        </li>
      </ul>
    </div>

    <div v-if="finalSummary">
      <h2 class="font-bold mt-6">🧠 Финальное саммари:</h2>
      <p class="mt-2">{{ finalSummary }}</p>
      <p class="text-sm text-gray-600">⏱ Время финальной генерации: {{ finalDuration }} сек</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from "vue";
import axios from "axios";

const text = ref("");
const loading = ref(false);
const tokenCount = ref(0);
const wordCount = ref(0);
const totalChunks = ref(0);
const chunkSummaries = ref([]);
const finalSummary = ref("");
const finalDuration = ref(0);

const params = ref({
  chunk_size: 1800,
  overlap: 200,
  temp_chunk: 0.4,
  temp_final: 0.6,
  chunk_prompt: "",
  final_prompt: ""
});

const processedChunks = computed(() => chunkSummaries.value.length);

function estimateTokens(text) {
  return Math.ceil(text.split(/\s+/).length * 1.3);
}

function estimateWord(text) {
  return Math.ceil(text.split(/\s+/).length);
}

const submitText = async () => {
  loading.value = true;
  chunkSummaries.value = [];
  finalSummary.value = "";
  finalDuration.value = 0;
  totalChunks.value = 0;

  try {
    tokenCount.value = estimateTokens(text.value);
    wordCount.value = estimateWord(text.value);
    const res = await axios.post("http://localhost:8000/summarize", {
      text: text.value,
      params: params.value
    });
    const taskId = res.data.task_id;

    const eventSource = new EventSource(`http://localhost:8000/stream/${taskId}`);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "chunk") {
        chunkSummaries.value.push(data);
        if (!totalChunks.value && data.total) {
          totalChunks.value = data.total;
        }
      }

      if (data.type === "final") {
        finalSummary.value = data.summary;
        finalDuration.value = data.duration;
        loading.value = false;
        eventSource.close();
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE error:", err);
      loading.value = false;
      eventSource.close();
    };
  } catch (err) {
    console.error("Ошибка:", err);
    loading.value = false;
  }
};
</script>

<style scoped>


textarea {
  width: 100%;
  padding: 0.5rem;
  font-family: inherit;
  border: 1px solid #ccc;
  border-radius: 4px;
}
button {
  padding: 0.5rem 1rem;
  background-color: #3b82f6;
  color: white;
  border: none;
  border-radius: 4px;
}
</style>
