import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:5000'; // Your Flask backend URL

// Function to fetch all recent pages
export const fetchAllPages = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/all`);
    return response.data;
  } catch (error) {
    console.error('Error fetching all pages:', error);
    throw error;
  }
};

// Function to fetch a specific page with its summary and paragraphs
export const fetchPageDetail = async (pageId) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/page/${pageId}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching page ${pageId}:`, error);
    throw error;
  }
};

// Function to perform semantic search
export const searchContent = async (query, top_k = 20, min_similarity = 0.0, from_date, to_date) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/search`, { query, top_k, min_similarity, from_date, to_date });
    return response.data;
  } catch (error) {
    console.error('Error searching content:', error);
    throw error;
  }
};

// Function to trigger page summarization
export const summarizePage = async (pageId) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/summarize_page`, { page_id: pageId });
    return response.data;
  } catch (error) {
    console.error(`Error summarizing page ${pageId}:`, error);
    throw error;
  }
};

// Function to get a summary for a specific query (does not store in DB)
export const summarizeQuery = async (query, top_k = 5) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/summarize_query`, { query, top_k });
    return response.data;
  } catch (error) {
    console.error(`Error summarizing query '${query}':`, error);
    throw error;
  }
};

// Function to get RAG answer
export const getRagAnswer = async (query, top_k = 5) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/rag_answer`, { query, top_k });
    return response.data;
  } catch (error) {
    console.error(`Error getting RAG answer for '${query}':`, error);
    throw error;
  }
};
