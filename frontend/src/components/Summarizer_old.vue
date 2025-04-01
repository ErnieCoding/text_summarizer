<template>
  <div class="p-4">
    <textarea v-model="text" placeholder="Введите текст" rows="8" class="w-full mb-4"></textarea>
    <button @click="submitText" :disabled="loading" class="mb-4 bg-blue-500 text-white px-4 py-2 rounded">
      Сделать саммари
    </button>

    <div v-if="loading">
      <p>⏳ Генерация саммари...</p>
    </div>

    <div v-if="totalChunks > 0" class="mb-2 text-sm text-gray-600">
      Обработка чанков: {{ processedChunks }} / {{ totalChunks }}
    </div>

    <div v-if="tokenCount > 0" class="mb-4 text-sm text-gray-500">
      Примерный размер текста: {{ tokenCount }} токенов
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
const totalChunks = ref(0);
const chunkSummaries = ref([]);
const finalSummary = ref("");
const finalDuration = ref(0);

const processedChunks = computed(() => chunkSummaries.value.length);

function estimateTokens(text) {
  return Math.ceil(text.split(/\s+/).length * 1.3);
}

const submitText = async () => {
  loading.value = true;
  chunkSummaries.value = [];
  finalSummary.value = "";
  finalDuration.value = 0;
  totalChunks.value = 0;

  try {
    tokenCount.value = estimateTokens(text.value);
    const res = await axios.post("http://localhost:8000/summarize", { text: text.value });
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