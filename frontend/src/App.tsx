import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import HomePage from "./pages/Home";
import LibraryPage from "./pages/Library";
import SearchPage from "./pages/Search";
import AskPage from "./pages/Ask";
import DocumentPage from "./pages/Document";
import AddPage from "./pages/Add";
import TopicsPage from "./pages/Topics";
import TopicDetailPage from "./pages/TopicDetail";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/library" element={<LibraryPage />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/ask" element={<AskPage />} />
        <Route path="/documents/:id" element={<DocumentPage />} />
        <Route path="/add" element={<AddPage />} />
        <Route path="/topics" element={<TopicsPage />} />
        <Route path="/topics/:topicSlug" element={<TopicDetailPage />} />
      </Route>
    </Routes>
  );
}
